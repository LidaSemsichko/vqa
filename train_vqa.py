import json
import os
import random

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
DATA_PATH = "data_collection/vqa_5k_en.json"
DATA_ROOT = "data_collection"
PROJECTOR_CKPT = "tiny_llava_projector_best.pth"
LORA_OUT_DIR = "tiny_llava_lora_best"
PROJECTOR_VQA_OUT = "tiny_llava_projector_vqa_best.pth"


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
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model.language_model = get_peft_model(model.language_model, lora_config)
    model.language_model.print_trainable_parameters()


def freeze_for_stage2(model):
    for param in model.vision_encoder.parameters():
        param.requires_grad = False
    for param in model.projector.parameters():
        param.requires_grad = True
    # LoRA params are already requires_grad=True; base LLM weights are frozen by PEFT


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
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            total_loss += outputs.loss.item() * pixel_values.size(0)
            count += pixel_values.size(0)
    return total_loss / count if count > 0 else float("nan")


def train_vqa(
    batch_size=4,
    grad_accum_steps=4,
    epochs=5,
    lr=2e-4,
    warmup_ratio=0.03,
    seed=42,
    max_grad_norm=1.0,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print("Loading VQA data...")
    all_records = load_vqa_data(DATA_PATH)
    train_records, val_records, test_records = split_data(all_records, seed=seed)
    print(f"Image-level split -> train: {len(train_records)}, val: {len(val_records)}, test: {len(test_records)}")

    train_ds = VQADataset(train_records, VISION_ID, TEXT_ID, data_root=DATA_ROOT)
    val_ds = VQADataset(val_records, VISION_ID, TEXT_ID, data_root=DATA_ROOT)
    test_ds = VQADataset(test_records, VISION_ID, TEXT_ID, data_root=DATA_ROOT)
    print(f"Sample-level split -> train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    print("Building model...")
    model = TinyLLaVA(vision_model_path=VISION_ID, text_model_path=TEXT_ID)

    train_ds.tokenizer.add_tokens(["<image>"])
    val_ds.tokenizer.add_tokens(["<image>"])
    test_ds.tokenizer.add_tokens(["<image>"])
    model.language_model.resize_token_embeddings(len(train_ds.tokenizer))
    model.image_token_id = train_ds.tokenizer.convert_tokens_to_ids("<image>")

    print(f"Loading projector weights from {PROJECTOR_CKPT}...")
    model.projector.load_state_dict(torch.load(PROJECTOR_CKPT, map_location="cpu", weights_only=True))

    print("Applying LoRA to language model...")
    apply_lora(model)
    freeze_for_stage2(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    model.to(device)

    collate = CollateWrapper(train_ds.tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=4, pin_memory=use_amp)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=4, pin_memory=use_amp)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=4, pin_memory=use_amp)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.01)
    total_steps = (len(train_loader) // grad_accum_steps) * epochs
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")
    print(f"\nTraining Stage 2 (LoRA + projector) for {epochs} epochs")
    print(f"Effective batch size: {batch_size * grad_accum_steps}, Steps: {total_steps}, Warmup: {warmup_steps}\n")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for step, batch in enumerate(loop):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            running_loss += loss.item() * grad_accum_steps
            loop.set_postfix(loss=f"{running_loss/(step+1):.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        train_loss = running_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, device, use_amp)
        print(f"Epoch {epoch+1}: train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.language_model.save_pretrained(LORA_OUT_DIR)
            torch.save(model.projector.state_dict(), PROJECTOR_VQA_OUT)
            print(f"  --> Saved best checkpoint (val={val_loss:.4f})")

    print("\nLoading best checkpoint for test evaluation...")
    model.language_model.load_adapter(LORA_OUT_DIR, "default")
    model.projector.load_state_dict(torch.load(PROJECTOR_VQA_OUT, weights_only=True))
    test_loss = evaluate(model, test_loader, device, use_amp)
    print(f"Final test loss: {test_loss:.4f}")
    print("Stage 2 training complete.")


if __name__ == "__main__":
    train_vqa()
