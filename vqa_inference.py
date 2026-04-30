import sys
import os
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, AutoTokenizer

from src.models import TinyLLaVA

# --- CONFIGURATION ---
VISION_ID = "openai/clip-vit-base-patch16"
TEXT_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Update this to the specific run you want to use (e.g., "runs/run0", "runs/run1")
RUN_DIR = "runs/run5"

# These paths are now relative to the RUN_DIR we created during training
LORA_DIR = os.path.join(RUN_DIR, "best_lora")
PROJECTOR_CKPT = os.path.join(RUN_DIR, "best_projector.pth")
# ---------------------

def load_model():
    # Quick sanity check
    if not os.path.exists(RUN_DIR):
        print(f"Error: Run directory '{RUN_DIR}' not found. Did you finish training?")
        sys.exit(1)

    print(f"Loading weights from {RUN_DIR}...")
    model = TinyLLaVA(vision_model_path=VISION_ID, text_model_path=TEXT_ID)

    tokenizer = AutoTokenizer.from_pretrained(TEXT_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.add_tokens(["<image>"])
    model.language_model.resize_token_embeddings(len(tokenizer))
    model.image_token_id = tokenizer.convert_tokens_to_ids("<image>")

    # Load the specific weights saved in the run directory
    model.projector.load_state_dict(torch.load(PROJECTOR_CKPT, map_location="cpu", weights_only=True))
    model.language_model = PeftModel.from_pretrained(model.language_model, LORA_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, tokenizer, AutoProcessor.from_pretrained(VISION_ID), device


@torch.no_grad()
def answer(model, tokenizer, processor, device, image_path, question):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    # Note: Ensure this prompt template matches exactly what you used in training!
    prompt = f"<|im_start|>user\n<image>\n{question}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    
    output_ids = model.generate(
        pixel_values=pixel_values,
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=64,
        do_sample=False,
        temperature=1.0,
        repetition_penalty=1.1,
    )

    # Decode while removing the prompt part
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Extract only the assistant's part if needed, or just return the stripped result
    return full_text.split("assistant\n")[-1].strip()


def main():
    if len(sys.argv) < 2:
        print("Usage: python vqa_inference.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    print("Loading model...")
    model, tokenizer, processor, device = load_model()
    print(f"Ready. Inference running on: {device}")
    print(f"Testing image: {image_path}\n")

    while True:
        try:
            question = input("Question (or Ctrl+C to quit): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
        if not question:
            continue
        
        response = answer(model, tokenizer, processor, device, image_path, question)
        print(f"Answer: {response}\n")


if __name__ == "__main__":
    main()