import torch

from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoTokenizer
from PIL import Image


class LLaVADataset(Dataset):
    def __init__(self, data_list, vision_model_path, text_model_path):
        """
        data_list: list of dicts like [{"image": "cat.jpg", "text": "A photo of a cat."}]
        """
        self.data = data_list

        self.image_processor = AutoProcessor.from_pretrained(vision_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        image = Image.open(item["image"]).convert("RGB")
        pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values.squeeze(0)

        text = item["text"]

        return {
            "pixel_values": pixel_values,
            "text": text
        }


class CollateWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        return collate_fn(batch, self.tokenizer)


def collate_fn(batch, tokenizer):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    texts = [item["text"] for item in batch]

    encoded_text = tokenizer(
        texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
    )

    input_ids = encoded_text["input_ids"]
    attention_mask = encoded_text["attention_mask"]

    labels = input_ids.clone()

    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
