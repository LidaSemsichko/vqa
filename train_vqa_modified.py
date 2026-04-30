import json
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from peft import LoraConfig, TaskType, get_peft_model
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from src.datasets import CollateWrapper, VQADataset
from src.models import TinyLLaVA


VISION_ID = "openai/clip-vit-base-patch16"
TEXT_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH = "data_collection/vqa_8k_en.json"
DATA_ROOT = "data_collection"
PROJECTOR_CKPT = "tiny_llava_projector_best.pth"
EVAL_STEPS = 500

def get_run_dir(base_dir="runs"):
    os.makedirs(base_dir, exist_ok=True)
    existing_runs = [d for d in os.listdir(base_dir) if d.startswith("run")]
    run_ids = [int(d.replace("run", "")) for d in existing_runs if d.replace("run", "").isdigit()]
    next_id = max(run_ids, default=-1) + 1
    run_dir = os.path.join(base_dir, f"run{next_id}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def save_stats(run_dir, stats, config):
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    with open(os.path.join(run_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=4)

    # Plotting
    plt.figure(figsize=(15, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    # Filter out None/NaN values for plotting
    train_steps, train_vals = zip(*stats["train_loss_history"])
    val_steps, val_vals = zip(*stats["val_loss_history"])
    
    plt.plot(train_steps, train_vals, label='Train Loss (Moving Avg)', alpha=0.6)
    plt.plot(val_steps, val_vals, label='Val Loss', marker='o', linewidth=2)
    plt.title('Training & Validation Loss')
    plt.xlabel('Optimization Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # Learning Rate Plot
    plt.subplot(1, 2, 2)
    lr_steps, lr_vals = zip(*stats["lr_history"])
    plt.plot(lr_steps, lr_vals, color='orange')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Optimization Steps')
    plt.ylabel('LR')
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "plots.png"))
    plt.close()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_vqa_data(path):
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return records

def split_data(records, train_ratio=0.8, val_ratio=0.1, seed=42):
    random.seed(seed)
    data = records.copy()
    random.shuffle(data)
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return data[:train_end], data[train_end:val_end], data[val_end:]

def apply_lora(model):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model.language_model = get_peft_model(model.language_model, lora_config)

def freeze_for_stage2(model):
    for param in model.vision_encoder.parameters():
        param.requires_grad = False
    for param in model.projector.parameters():
        param.requires_grad = True

def evaluate(model, dataloader, device, use_amp):
    model.eval()
    total_loss, count = 0.0, 0
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item() * pixel_values.size(0)
            count += pixel_values.size(0)
    model.train()
    return total_loss / count if count > 0 else float("nan")

def train_vqa(
    batch_size=8,
    grad_accum_steps=2,
    epochs=5,
    lr=2e-5,
    warmup_ratio=0.1,
    seed=42,
    max_grad_norm=1.0,
):
    set_seed(seed)
    run_dir = get_run_dir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    # Data loading
    all_records = load_vqa_data(DATA_PATH)
    train_records, val_records, test_records = split_data(all_records, seed=seed)
    train_ds = VQADataset(train_records, VISION_ID, TEXT_ID, data_root=DATA_ROOT)
    val_ds = VQADataset(val_records, VISION_ID, TEXT_ID, data_root=DATA_ROOT)
    test_ds = VQADataset(test_records, VISION_ID, TEXT_ID, data_root=DATA_ROOT)

    # Model setup — add <image> token to ALL tokenizers before any dataset uses them
    model = TinyLLaVA(vision_model_path=VISION_ID, text_model_path=TEXT_ID)
    for ds in (train_ds, val_ds, test_ds):
        ds.tokenizer.add_tokens(["<image>"])
    model.language_model.resize_token_embeddings(len(train_ds.tokenizer))
    model.image_token_id = train_ds.tokenizer.convert_tokens_to_ids("<image>")
    model.projector.load_state_dict(torch.load(PROJECTOR_CKPT, map_location="cpu", weights_only=True))
    apply_lora(model)
    freeze_for_stage2(model)
    model.to(device)

    # Dataloaders
    collate = CollateWrapper(train_ds.tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)

    # Training components
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.05)
    total_opt_steps = (len(train_loader) // grad_accum_steps) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_opt_steps * warmup_ratio), num_training_steps=total_opt_steps)
    scaler = GradScaler("cuda", enabled=use_amp)

    # Tracking variables
    stats = {"train_loss_history": [], "val_loss_history": [], "lr_history": []}
    best_val_loss = float("inf")
    global_step = 0
    running_loss_since_eval = 0.0
    steps_since_eval = 0

    print(f"\nTraining for {epochs} epochs. Evaluation every {EVAL_STEPS} steps.")

    for epoch in range(epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for step, batch in enumerate(loop):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / grad_accum_steps

            scaler.scale(loss).backward()
            running_loss_since_eval += loss.item() * grad_accum_steps
            
            # Optimization step
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                
                global_step += 1
                steps_since_eval += 1
                
                # Check for Step-based Evaluation
                if global_step % EVAL_STEPS == 0:
                    avg_train_loss = running_loss_since_eval / (steps_since_eval * grad_accum_steps)
                    val_loss = evaluate(model, val_loader, device, use_amp)

                    stats["train_loss_history"].append((global_step, avg_train_loss))
                    stats["val_loss_history"].append((global_step, val_loss))
                    stats["lr_history"].append((global_step, scheduler.get_last_lr()[0]))

                    print(f"\n[Step {global_step}] Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        model.language_model.save_pretrained(os.path.join(run_dir, "best_lora"))
                        torch.save(model.projector.state_dict(), os.path.join(run_dir, "best_projector.pth"))

                    # Reset local counters
                    running_loss_since_eval = 0.0
                    steps_since_eval = 0

            loop.set_postfix(loss=f"{loss.item()*grad_accum_steps:.4f}", step=global_step)

        # Epoch-end Evaluation (if not already evaluated at this exact step)
        if global_step % EVAL_STEPS != 0:
            avg_train_loss = running_loss_since_eval / max(1, steps_since_eval * grad_accum_steps)
            val_loss = evaluate(model, val_loader, device, use_amp)
            
            stats["train_loss_history"].append((global_step, avg_train_loss))
            stats["val_loss_history"].append((global_step, val_loss))
            stats["lr_history"].append((global_step, scheduler.get_last_lr()[0]))
            
            print(f"--- Epoch {epoch+1} End: Train={avg_train_loss:.4f} Val={val_loss:.4f} ---")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.language_model.save_pretrained(os.path.join(run_dir, "best_lora"))
                torch.save(model.projector.state_dict(), os.path.join(run_dir, "best_projector.pth"))

            running_loss_since_eval = 0.0
            steps_since_eval = 0

    # Final logic
    print("\nTraining complete. Loading best and testing...")
    model.language_model.load_adapter(os.path.join(run_dir, "best_lora"), "default")
    test_loss = evaluate(model, test_loader, device, use_amp)
    stats["test_loss"] = test_loss
    
    config = {"batch_size": batch_size, "lr": lr, "epochs": epochs, "eval_steps": EVAL_STEPS}
    save_stats(run_dir, stats, config)

if __name__ == "__main__":
    train_vqa()