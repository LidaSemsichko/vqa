import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM

class TinyLLaVA(nn.Module):
    def __init__(
        self,
        vision_model_path="openai/clip-vit-base-patch16",
        text_model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ):
        super().__init__()

        self.vision_encoder = AutoModel.from_pretrained(vision_model_path).vision_model

        self.language_model = AutoModelForCausalLM.from_pretrained(text_model_path)

        vision_dim = self.vision_encoder.config.hidden_size
        text_dim = self.language_model.config.hidden_size

        self.projector = nn.Sequential(
            nn.Linear(vision_dim, text_dim), nn.GELU(), nn.Linear(text_dim, text_dim)
        )

    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):

        vision_outputs = self.vision_encoder(pixel_values).last_hidden_state

        projected_vision = self.projector(vision_outputs)

        text_embeddings = self.language_model.get_input_embeddings()(input_ids)

        combined_embeddings = torch.cat([projected_vision, text_embeddings], dim=1)

        if attention_mask is not None:

            vision_mask = torch.ones(
                (attention_mask.shape[0], projected_vision.shape[1]),
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )

            combined_mask = torch.cat([vision_mask, attention_mask], dim=1)
        else:
            combined_mask = None

        if labels is not None:
            vision_labels = torch.full(
                (labels.shape[0], projected_vision.shape[1]),
                -100,
                device=labels.device,
                dtype=labels.dtype,
            )
            combined_labels = torch.cat([vision_labels, labels], dim=1)
        else:
            combined_labels = None

        return self.language_model(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_mask,
            labels=combined_labels,
        )

    @torch.no_grad()
    def generate(self, pixel_values, input_ids, attention_mask=None, **kwargs):

        vision_outputs = self.vision_encoder(pixel_values).last_hidden_state
        projected_vision = self.projector(vision_outputs)

        text_embeddings = self.language_model.get_input_embeddings()(input_ids)

        combined_embeddings = torch.cat([projected_vision, text_embeddings], dim=1)

        if attention_mask is not None:
            vision_mask = torch.ones(
                (attention_mask.shape[0], projected_vision.shape[1]),
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            combined_mask = torch.cat([vision_mask, attention_mask], dim=1)
        else:
            combined_mask = None

        return self.language_model.generate(
            inputs_embeds=combined_embeddings, attention_mask=combined_mask, **kwargs
        )
