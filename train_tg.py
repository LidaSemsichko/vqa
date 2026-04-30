import argparse
import json
import os
import random

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from peft import LoraConfig, TaskType, get_peft_model, set_peft_model_state_dict
from PIL import Image
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from src.models import TinyLLaVA


VISION_ID       = "openai/clip-vit-base-patch16"
TEXT_ID         = "Qwen/Qwen2.5-0.5B-Instruct"
PROJECTOR_CKPT  = "tiny_llava_projector_vqa_best.pth"
DATA_PATH       = "tg_data/data_collection/vqa_tg.json"
DATA_ROOT       = "tg_data/data_collection"
EVAL_STEPS      = 200


# ── Dataset ───────────────────────────────────────────────────────────────────

class TelegramDataset(Dataset):
    """
    Each Telegram item has up to 3 conversations (list of message dicts).
    We expand them into individual training samples.

    Mode 'with_description':
        Prompt = <image> + post caption → conversation text
    Mode 'no_description':
        Prompt = <image> → conversation text
    """

    def __init__(
        self,
        data_list: list,
        vision_model_path: str,
        text_model_path: str,
        data_root: str = "tg_data/data_collection",
        use_description: bool = False,
        max_seq_len: int = 512,
    ):
        self.use_description = use_description
        self.max_seq_len     = max_seq_len
        self.samples: list[dict] = []

        for item in data_list:
            img_path = os.path.join(data_root, item["image"])
            caption  = item.get("caption", "").strip()

            for conv in item.get("conversations", []):
                if not conv:
                    continue
                conv_text = "\n".join(
                    f"{msg['sender']}: {msg['text']}" for msg in conv
                )
                self.samples.append({
                    "image":        img_path,
                    "caption":      caption,
                    "conversation": conv_text,
                })

        self.image_processor = AutoProcessor.from_pretrained(vision_model_path)
        self.tokenizer       = AutoTokenizer.from_pretrained(text_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        item = self.samples[idx]

        image        = Image.open(item["image"]).convert("RGB")
        pixel_values = self.image_processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        if self.use_description and item["caption"]:
            desc_part = f"Description: {item['caption']}\n"
        else:
            desc_part = ""

        prompt = (
            f"<|im_start|>user\n<image>\n{desc_part}"
            f"Discuss this image:<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        answer = item["conversation"] + "<|im_end|>"

        prompt_ids = self.tokenizer(prompt, add_special_tokens=True).input_ids
        answer_ids = self.tokenizer(answer, add_special_tokens=False).input_ids

        # Truncate answer if combined length exceeds max_seq_len
        max_answer = self.max_seq_len - len(prompt_ids)
        if max_answer <= 0:
            answer_ids = []
        elif len(answer_ids) > max_answer:
            answer_ids = answer_ids[:max_answer]

        input_ids = prompt_ids + answer_ids
        labels    = [-100] * len(prompt_ids) + answer_ids

        return {
            "pixel_values": pixel_values,
            "input_ids":    torch.tensor(input_ids, dtype=torch.long),
            "labels":       torch.tensor(labels,    dtype=torch.long),
        }


# ── Collate ───────────────────────────────────────────────────────────────────

def collate_fn(batch: list, pad_token_id: int) -> dict:
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [b["input_ids"] for b in batch], batch_first=True, padding_value=pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [b["labels"] for b in batch], batch_first=True, padding_value=-100
    )
    attention_mask = input_ids.ne(pad_token_id).long()
    return {"pixel_values": pixel_values, "input_ids": input_ids,
            "attention_mask": attention_mask, "labels": labels}


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_run_dir(base_dir: str = "runs") -> str:
    os.makedirs(base_dir, exist_ok=True)
    existing = [d for d in os.listdir(base_dir) if d.startswith("run")]
    ids = [int(d[3:]) for d in existing if d[3:].isdigit()]
    run_dir = os.path.join(base_dir, f"run{max(ids, default=-1)+1}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def split_data(records: list, train_ratio=0.8, val_ratio=0.1, seed=42):
    random.seed(seed)
    data = records.copy()
    random.shuffle(data)
    n = len(data)
    t = int(n * train_ratio)
    v = t + int(n * val_ratio)
    return data[:t], data[t:v], data[v:]


def apply_lora(model: TinyLLaVA) -> None:
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model.language_model = get_peft_model(model.language_model, lora_cfg)


def freeze_vision(model: TinyLLaVA) -> None:
    for p in model.vision_encoder.parameters():
        p.requires_grad = False
    for p in model.projector.parameters():
        p.requires_grad = True


def enable_gradient_checkpointing(model: TinyLLaVA) -> None:
    """Recompute activations during backward pass to save GPU memory."""
    model.language_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )


def evaluate(model: TinyLLaVA, loader: DataLoader, device: torch.device,
             use_amp: bool, amp_dtype: torch.dtype) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            pv  = batch["pixel_values"].to(device)
            ids = batch["input_ids"].to(device)
            msk = batch["attention_mask"].to(device)
            lbl = batch["labels"].to(device)
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(pixel_values=pv, input_ids=ids, attention_mask=msk, labels=lbl)
            total += out.loss.item() * pv.size(0)
            count += pv.size(0)
    model.train()
    return total / count if count else float("nan")


def save_stats(run_dir: str, stats: dict, config: dict) -> None:
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)
    with open(os.path.join(run_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=4)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    ts, tv = zip(*stats["train_loss_history"])
    vs, vv = zip(*stats["val_loss_history"])
    plt.plot(ts, tv, label="Train (moving avg)", alpha=0.6)
    plt.plot(vs, vv, label="Val", marker="o", linewidth=2)
    plt.title("Loss"); plt.xlabel("Steps"); plt.legend(); plt.grid(True, linestyle="--", alpha=0.7)

    plt.subplot(1, 2, 2)
    ls, lv = zip(*stats["lr_history"])
    plt.plot(ls, lv, color="orange")
    plt.title("Learning Rate"); plt.xlabel("Steps"); plt.yscale("log"); plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "plots.png"))
    plt.close()


# ── Training ──────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    run_dir    = get_run_dir()
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp    = device.type == "cuda"
    use_desc   = args.mode == "with_description"

    # Pick the best available low-precision dtype.
    # bfloat16: same exponent range as fp32, no GradScaler needed.
    # float16:  smaller range, needs GradScaler.
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    elif device.type == "cuda":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32

    print(f"Run dir : {run_dir}")
    print(f"Mode    : {args.mode}")
    print(f"Device  : {device}")
    print(f"Dtype   : {amp_dtype}")

    # Data
    with open(args.data_path, "r", encoding="utf-8") as f:
        all_records = json.load(f)

    train_rec, val_rec, test_rec = split_data(all_records, seed=args.seed)

    def make_ds(records):
        return TelegramDataset(
            records, VISION_ID, TEXT_ID,
            data_root=args.data_root,
            use_description=use_desc,
            max_seq_len=args.max_seq_len,
        )

    train_ds = make_ds(train_rec)
    val_ds   = make_ds(val_rec)
    test_ds  = make_ds(test_rec)

    print(f"Samples — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    # Model
    model = TinyLLaVA(vision_model_path=VISION_ID, text_model_path=TEXT_ID)
    tokenizer = train_ds.tokenizer
    tokenizer.add_tokens(["<image>"])
    for ds in (val_ds, test_ds):
        ds.tokenizer.add_tokens(["<image>"])

    model.language_model.resize_token_embeddings(len(tokenizer))
    model.image_token_id = tokenizer.convert_tokens_to_ids("<image>")

    ckpt_path = args.projector_ckpt
    print(f"Loading projector from: {ckpt_path}")
    model.projector.load_state_dict(
        torch.load(ckpt_path, map_location="cpu", weights_only=True)
    )

    apply_lora(model)
    freeze_vision(model)
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(model)

    # Convert weights to bf16/fp16 BEFORE moving to GPU — halves VRAM for weights.
    if amp_dtype != torch.float32:
        model = model.to(amp_dtype)
    model.to(device)

    # Dataloaders
    pad_id = tokenizer.pad_token_id
    collate = lambda batch: collate_fn(batch, pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=4, pin_memory=use_amp)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate, num_workers=2)

    # Optimizer / scheduler / scaler
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.05,
    )
    total_opt_steps = (len(train_loader) // args.grad_accum) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_opt_steps * args.warmup_ratio),
        num_training_steps=total_opt_steps,
    )
    # GradScaler only needed for float16 (bf16 has fp32-range exponents — no underflow).
    use_scaler = use_amp and (amp_dtype == torch.float16)
    scaler = GradScaler("cuda", enabled=use_scaler)

    stats = {"train_loss_history": [], "val_loss_history": [], "lr_history": []}
    best_val_loss = float("inf")
    global_step   = 0
    running_loss  = 0.0
    steps_since   = 0

    print(f"\nTraining {args.epochs} epochs, eval every {EVAL_STEPS} steps.\n")

    for epoch in range(args.epochs):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for step, batch in enumerate(loop):
            pv  = batch["pixel_values"].to(device)
            ids = batch["input_ids"].to(device)
            msk = batch["attention_mask"].to(device)
            lbl = batch["labels"].to(device)

            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out  = model(pixel_values=pv, input_ids=ids, attention_mask=msk, labels=lbl)
                loss = out.loss / args.grad_accum

            scaler.scale(loss).backward()
            running_loss += loss.item() * args.grad_accum

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

                global_step += 1
                steps_since += 1

                if global_step % EVAL_STEPS == 0:
                    avg_train = running_loss / (steps_since * args.grad_accum)
                    val_loss  = evaluate(model, val_loader, device, use_amp, amp_dtype)

                    stats["train_loss_history"].append((global_step, avg_train))
                    stats["val_loss_history"].append((global_step, val_loss))
                    stats["lr_history"].append((global_step, scheduler.get_last_lr()[0]))

                    print(f"\n[Step {global_step}] train={avg_train:.4f}  val={val_loss:.4f}")

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        model.language_model.save_pretrained(os.path.join(run_dir, "best_lora"))
                        torch.save(model.projector.state_dict(),
                                   os.path.join(run_dir, "best_projector.pth"))

                    running_loss = 0.0
                    steps_since  = 0

            loop.set_postfix(loss=f"{loss.item()*args.grad_accum:.4f}", step=global_step)

        # Epoch-end eval
        if global_step % EVAL_STEPS != 0:
            avg_train = running_loss / max(1, steps_since * args.grad_accum)
            val_loss  = evaluate(model, val_loader, device, use_amp, amp_dtype)

            stats["train_loss_history"].append((global_step, avg_train))
            stats["val_loss_history"].append((global_step, val_loss))
            stats["lr_history"].append((global_step, scheduler.get_last_lr()[0]))

            print(f"--- Epoch {epoch+1} end: train={avg_train:.4f}  val={val_loss:.4f} ---")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.language_model.save_pretrained(os.path.join(run_dir, "best_lora"))
                torch.save(model.projector.state_dict(),
                           os.path.join(run_dir, "best_projector.pth"))

            running_loss = 0.0
            steps_since  = 0

    # Final test — load best adapter weights without going through HF Hub validation
    print("\nLoading best checkpoint for test evaluation...")
    best_lora_dir = os.path.join(run_dir, "best_lora")
    sf_path  = os.path.join(best_lora_dir, "adapter_model.safetensors")
    bin_path = os.path.join(best_lora_dir, "adapter_model.bin")
    if os.path.exists(sf_path):
        from safetensors.torch import load_file
        set_peft_model_state_dict(model.language_model, load_file(sf_path))
    elif os.path.exists(bin_path):
        set_peft_model_state_dict(model.language_model,
                                  torch.load(bin_path, map_location="cpu", weights_only=True))
    else:
        print("  Warning: no saved adapter found, evaluating with last-epoch weights.")
    test_loss = evaluate(model, test_loader, device, use_amp, amp_dtype)
    stats["test_loss"] = test_loss
    print(f"Test loss: {test_loss:.4f}")

    config = vars(args)
    save_stats(run_dir, stats, config)
    print(f"\nDone. Results saved to {run_dir}/")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train TinyLLaVA on Telegram VQA dataset.")
    ap.add_argument(
        "--mode", required=True,
        choices=["with_description", "no_description"],
        help=(
            "with_description: image + post caption as context → conversation; "
            "no_description:   image only → conversation"
        ),
    )
    ap.add_argument("--data_path",      default=DATA_PATH,      help="Path to vqa_tg.json")
    ap.add_argument("--data_root",      default=DATA_ROOT,      help="Root folder for images")
    ap.add_argument("--projector_ckpt", default=PROJECTOR_CKPT, help="Projector checkpoint .pth")
    ap.add_argument("--epochs",         type=int,   default=5)
    ap.add_argument("--batch_size",     type=int,   default=2,
                    help="Per-GPU batch size (default 4; reduce to 2 if still OOM)")
    ap.add_argument("--grad_accum",     type=int,   default=8,
                    help="Gradient accumulation steps (effective batch = batch_size × grad_accum)")
    ap.add_argument("--lr",             type=float, default=2e-5)
    ap.add_argument("--warmup_ratio",   type=float, default=0.1)
    ap.add_argument("--max_grad_norm",  type=float, default=1.0)
    ap.add_argument("--max_seq_len",    type=int,   default=256,
                    help="Max tokens per sample. Telegram messages are short; 128 covers most.")
    ap.add_argument("--gradient_checkpointing", action="store_true", default=True,
                    help="Recompute activations to save GPU memory (default: on)")
    ap.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing",
                    action="store_false", help="Disable gradient checkpointing")
    ap.add_argument("--seed",           type=int,   default=42)
    return ap.parse_args()


if __name__ == "__main__":
    train(parse_args())
