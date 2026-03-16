import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
from src.models import TinyLLaVA

def run_inference(image_path,
                  projector_weights_path,
                  vision_id="openai/clip-vit-base-patch16",
                  text_id="Qwen/Qwen2.5-0.5B-Instruct"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Initializing model...")
    model = TinyLLaVA(vision_model_path=vision_id, text_model_path=text_id)
    model.projector.load_state_dict(torch.load(projector_weights_path, map_location=device))

    processor = AutoProcessor.from_pretrained(vision_id)
    tokenizer = AutoTokenizer.from_pretrained(text_id)
    tokenizer.add_tokens(["<image>"], special_tokens = True)
    model.language_model.resize_token_embeddings(len(tokenizer))
    model.image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    
    model.to(device).to(torch.float32)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

    prompt = "<|im_start|>user\n<image>\nDescribe this image in detail.<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print("Generating description...")

    output_ids = None

    with torch.no_grad():
        output_ids = model.generate(
            pixel_values=pixel_values,
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.7,
        )

    generated_text = tokenizer.decode(output_ids[0])#, skip_special_tokens=True)
    print("-" * 30)
    print(f"IMAGE: {image_path}")
    print(f"MODEL: {generated_text.strip()}")
    print("-" * 30)

if __name__ == "__main__":

    run_inference(
        image_path="flickr30k_inference/Images/boba1.png", 
        projector_weights_path="tiny_llava_projector_best.pth"
    )
