import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM

class TinyLLaVA(nn.Module):
    def __init__(
        self,
        vision_model_path="openai/clip-vit-base-patch16",
        text_model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        image_token_id=32000,
    ):
        super().__init__()
        self.image_token_id = image_token_id

        self.vision_encoder = AutoModel.from_pretrained(vision_model_path).vision_model
        self.language_model = AutoModelForCausalLM.from_pretrained(text_model_path)

        vision_dim = self.vision_encoder.config.hidden_size
        text_dim = self.language_model.config.hidden_size

        self.projector = nn.Sequential(
            nn.Linear(vision_dim, text_dim),
            nn.GELU(),
            nn.Linear(text_dim, text_dim)
        )

    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
        vision_outputs = self.vision_encoder(pixel_values).last_hidden_state
        projected_vision = self.projector(vision_outputs)
        text_embeddings = self.language_model.get_input_embeddings()(input_ids)

        batch_size = input_ids.shape[0]
        new_embeds = []
        new_labels = [] if labels is not None else None
        new_masks = [] if attention_mask is not None else None

        for i in range(batch_size):

            image_indices = (input_ids[i] == self.image_token_id).nonzero(as_tuple=True)[0]

            if image_indices.numel() == 0:
                
                new_embeds.append(text_embeddings[i])
                if labels is not None:
                    new_labels.append(labels[i])
                if attention_mask is not None:
                    new_masks.append(attention_mask[i])

            else:

                idx = image_indices[0]

                embeds_before = text_embeddings[i, :idx]
                embeds_after = text_embeddings[i, idx+1:]
                concat_embeds = torch.cat([embeds_before, projected_vision[i], embeds_after], dim=0)
                new_embeds.append(concat_embeds)

                if labels is not None:
                    labels_before = labels[i, :idx]
                    labels_image = torch.full((projected_vision.shape[1],), -100, dtype=labels.dtype, device=labels.device)
                    labels_after = labels[i, idx+1:]
                    concat_labels = torch.cat([labels_before, labels_image, labels_after], dim=0)
                    new_labels.append(concat_labels)

                if attention_mask is not None:
                    mask_before = attention_mask[i, :idx]
                    mask_image = torch.ones((projected_vision.shape[1],), dtype=attention_mask.dtype, device=attention_mask.device)
                    mask_after = attention_mask[i, idx+1:]
                    concat_mask = torch.cat([mask_before, mask_image, mask_after], dim=0)
                    new_masks.append(concat_mask)

        inputs_embeds = torch.stack(new_embeds, dim=0)

        if labels is not None:
            labels = torch.stack(new_labels, dim=0)
        if attention_mask is not None:
            attention_mask = torch.stack(new_masks, dim=0)

        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(self, pixel_values, input_ids, attention_mask=None, **kwargs):
        vision_outputs = self.vision_encoder(pixel_values).last_hidden_state
        projected_vision = self.projector(vision_outputs)
        text_embeddings = self.language_model.get_input_embeddings()(input_ids)

        batch_size = input_ids.shape[0]
        new_embeds = []
        new_masks = []

        for i in range(batch_size):

            image_idx = (input_ids[i] == self.image_token_id).nonzero(as_tuple=True)[0]

            idx = image_idx[0]
            embeds_before = text_embeddings[i, :idx]
            embeds_after = text_embeddings[i, idx+1:]
            new_embeds.append(torch.cat([embeds_before, projected_vision[i], embeds_after], dim=0))

            if attention_mask is not None:
                mask_before = attention_mask[i, :idx]
                mask_image = torch.ones((projected_vision.shape[1],), dtype=attention_mask.dtype, device=attention_mask.device)
                mask_after = attention_mask[i, idx+1:]
                new_masks.append(torch.cat([mask_before, mask_image, mask_after], dim=0))

        inputs_embeds = torch.stack(new_embeds, dim=0)
        attention_mask = torch.stack(new_masks, dim=0) if attention_mask is not None else None

        res = self.language_model.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs
        )

        return res