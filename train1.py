import os
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from torch.amp import autocast, GradScaler

from src.datasets import LLaVADataset, CollateWrapper
from src.models import TinyLLaVA


def set_seed(seed=42):
    """Фіксує всі генератори випадкових чисел для повної відтворюваності."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_data(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config path not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        lines = f.readlines()[1:]

    records = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        parts = line.split(",", maxsplit=1)
        if len(parts) != 2:
            print(f"Skipping malformed line {i+2}: {line}")
            continue
        image_name, text = parts
        text = text.strip()
        if not text:
            print(f"Skipping empty text at line {i+2}: {line}")
            continue
        image_path = os.path.join("flickr30k", "Images", image_name.strip())
        records.append({"image": image_path, "text": text})

    if len(records) == 0:
        raise ValueError("No data found in captions file.")
    return records


def purity_check(records):
    if len(records) == 0:
        raise ValueError("Purity check failed: dataset is empty.")

    seen_pairs = set()
    seen_images = set()
    cleaned_records = []
    dup_count = 0

    for idx, rec in enumerate(records):
        if "image" not in rec or "text" not in rec:
            raise ValueError(
                f"Purity check failed: missing keys in record {idx}: {rec}"
            )
        if not rec["text"] or not isinstance(rec["text"], str):
            raise ValueError(f"Purity check failed: invalid text at record {idx}")
        image_path = rec["image"]
        pair = (image_path, rec["text"])
        if pair in seen_pairs:
            dup_count += 1
            continue

        if not os.path.exists(image_path):
            raise FileNotFoundError(
                f"Purity check failed: image not found at {image_path}"
            )

        seen_pairs.add(pair)
        seen_images.add(image_path)
        cleaned_records.append(rec)

    print(
        f"Purity check passed: {len(cleaned_records)} records, unique images = {len(seen_images)}, unique image-text pairs = {len(seen_pairs)}, duplicates removed = {dup_count}"
    )
    return cleaned_records


def split_data(records, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    # ... (твій код без змін) ...
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    total = len(records)
    if total < 3:
        raise ValueError("Need at least 3 examples to split train/val/test")

    random.seed(seed)
    records_copy = records.copy()
    random.shuffle(records_copy)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_data = records_copy[:train_end]
    val_data = records_copy[train_end:val_end]
    test_data = records_copy[val_end:]

    if len(train_data) == 0 or len(val_data) == 0 or len(test_data) == 0:
        raise ValueError(
            "One of the splits is empty. Reduce split ratios or add more data."
        )

    return train_data, val_data, test_data


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Використовуємо autocast для оцінки, щоб не отримати OOM
            with autocast(device_type=device.type, dtype=torch.float16):
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            
            total_loss += outputs.loss.item() * pixel_values.size(0)
            count += pixel_values.size(0)

    if count == 0:
        return float("nan")
    return total_loss / count


def freeze_weights(model):
    for param in model.vision_encoder.parameters():
        param.requires_grad = False
    for param in model.language_model.parameters():
        param.requires_grad = False

    for param in model.projector.parameters():
        param.requires_grad = True

    if hasattr(model.language_model, "enable_input_require_grads"):
        model.language_model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.language_model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


def train_llava_projector(
    config_path="flickr30k/captions.txt",
    vision_id="openai/clip-vit-base-patch16",
    text_id="Qwen/Qwen2.5-0.5B-Instruct",
    batch_size=1,
    grad_accum_steps=2,
    epochs=20,
    lr=4e-4,
    warmup_ratio=0.1,
    seed=42,
    max_grad_norm=1.0
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print("Loading data...")
    all_records = get_data(config_path)
    all_records = purity_check(all_records)

    train_records, val_records, test_records = split_data(all_records, seed=seed)
    print(
        f"Split sizes -> train: {len(train_records)}, val: {len(val_records)}, test: {len(test_records)}"
    )

    train_ds = LLaVADataset(train_records, vision_id, text_id)
    val_ds = LLaVADataset(val_records, vision_id, text_id)
    test_ds = LLaVADataset(test_records, vision_id, text_id)

    print("Loading model...")
    
    model = TinyLLaVA(vision_model_path=vision_id, text_model_path=text_id)
    train_ds.tokenizer.add_tokens(["<image>"])
    val_ds.tokenizer.add_tokens(["<image>"])
    test_ds.tokenizer.add_tokens(["<image>"])

    model.language_model.resize_token_embeddings(len(train_ds.tokenizer))
    model.image_token_id = train_ds.tokenizer.convert_tokens_to_ids("<image>") # Todo: add general tockenizer
    model.to(device)
    freeze_weights(model)

    collate = CollateWrapper(train_ds.tokenizer)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate, num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate, num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate, num_workers=4,
        pin_memory=torch.cuda.is_available()
    )


    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (projector only): {trainable_params:,}")

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    total_steps = (len(train_loader) // grad_accum_steps) * epochs
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    scaler = GradScaler('cuda', enabled=use_amp)
    best_val_loss = float('inf')

    effective_batch_size = batch_size * grad_accum_steps
    print(f"Training for {epochs} epochs (Effective Batch Size: {effective_batch_size})")
    print(f"Total optimization steps: {total_steps}, Warmup steps: {warmup_steps}, Peak LR: {lr}")

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

            current_lr = scheduler.get_last_lr()[0]

            loop.set_postfix(
                loss=f"{(running_loss / (step + 1)):.4f}", 
                lr=f"{current_lr:.2e}"
            )

        train_loss = running_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1} finished: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(model.projector.state_dict(), "tiny_llava_projector_best.pth")
            print(f"--> New best model saved (val_loss: {val_loss:.4f})")

    print("Loading best model for final evaluation...")
    model.projector.load_state_dict(torch.load("tiny_llava_projector_best.pth", weights_only=True))
    test_loss = evaluate(model, test_loader, device)
    print(f"Final test loss: {test_loss:.4f}")

    print("Training complete.")


if __name__ == "__main__":
    train_llava_projector()