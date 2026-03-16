import torch

from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoTokenizer
from PIL import Image


class LLaVADataset(Dataset):
    def __init__(self, data_list, vision_model_path, text_model_path):

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

        prompt = "<|im_start|>user\n<image>\nDescribe this image in detail.<|im_end|>\n<|im_start|>assistant\n"
        answer = item["text"] + "<|im_end|>" 

        prompt_ids = self.tokenizer(prompt, add_special_tokens=True).input_ids
        answer_ids = self.tokenizer(answer, add_special_tokens=False).input_ids

        input_ids = prompt_ids + answer_ids

        labels = [-100] * len(prompt_ids) + answer_ids  

        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }


class CollateWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        return collate_fn(batch, self.tokenizer)



def collate_fn(batch, tokenizer):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])

    input_ids_list = [item["input_ids"] for item in batch]
    labels_list = [item["labels"] for item in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id
    )

    labels = torch.nn.utils.rnn.pad_sequence(
        labels_list, batch_first=True, padding_value=-100
    )

    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()

    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
