import torch
import torch.nn as nn

from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
import vilt.modules.vision_transformer as vit
from vilt.modules import heads, vilt_utils

class GQAViLT(nn.Module):
    def __init__(self, num_answers):
        super().__init__()
        self.num_answers = num_answers

        # default parameters
        bert_config = BertConfig(
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=768 * 4,
            max_position_embeddings=40,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )

        self.text_embeddings = BertEmbeddings(bert_config)
        self.text_embeddings.apply(vilt_utils.init_weights)

        self.token_type_embeddings = nn.Embedding(2, 768)
        self.token_type_embeddings.apply(vilt_utils.init_weights)

        self.transformer = getattr(vit, "vit_base_patch32_384")(
            pretrained=False, drop_rate=0.1
        )
        
        self.pooler = heads.Pooler(768)
        self.pooler.apply(vilt_utils.init_weights)

        self.vqa_classifier = nn.Sequential(
            nn.Linear(768, 768 * 2),
            nn.LayerNorm(768 * 2),
            nn.GELU(),
            nn.Linear(768 * 2, num_answers),
        )
        self.vqa_classifier.apply(vilt_utils.init_weights)
    
    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        do_mlm = "_mlm" if mask_text else ""
        text_ids = batch[f"text_ids{do_mlm}"]
        text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        text_embeds = self.text_embeddings(text_ids)

        if image_embeds is None and image_masks is None:
            img = batch[imgkey][0]
            (
                image_embeds,
                image_masks,
                patch_index,
                image_labels,
            ) = self.transformer.visual_embed(
                img,
                max_image_len=-1,
                mask_it=mask_image,
            )
        else:
            patch_index, image_labels = (
                None,
                None,
            )

        text_embeds, image_embeds = (
            text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks)),
            image_embeds
            + self.token_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )

        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        co_masks = torch.cat([text_masks, image_masks], dim=1)

        x = co_embeds

        for i, blk in enumerate(self.transformer.blocks):
            x, _attn = blk(x, mask=co_masks)

        x = self.transformer.norm(x)
        text_feats, image_feats = (
            x[:, : text_embeds.shape[1]],
            x[:, text_embeds.shape[1] :],
        )
        cls_feats = self.pooler(x)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "image_labels": image_labels,
            "image_masks": image_masks,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "patch_index": patch_index,
        }

        return ret
    
    def forward(self, batch):
        infer = self.infer(batch, mask_text=False, mask_image=False)
        vqa_logits = self.vqa_classifier(infer['cls_feats'])
        return vqa_logits