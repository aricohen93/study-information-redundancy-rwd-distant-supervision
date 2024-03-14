from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer

from oeciml.torchmodules.classification_head import (
    RobertaClassificationHead,
    SimpleClassificationHead,
)

from ..registry import registry
from .helpers.misc import shift


@registry.model("tokenizer")
def tokenizer_from_pretrained(
    path: Union[str, Path],
    additional_special_tokens: Optional[List[str]] = None,
):
    return AutoTokenizer.from_pretrained(
        path,
        additional_special_tokens=additional_special_tokens,
    )


@registry.model("transformer")
def transformer_from_pretrained(path: Union[str, Path]):
    return AutoModel.from_pretrained(path)


class Pooler(torch.nn.Module):
    def __init__(
        self,
        mode="mean",
        dropout_p=0.0,
        input_size=None,
        n_heads=None,
        do_value_proj=False,
    ):
        super().__init__()
        self.mode = mode
        assert mode in ("max", "sum", "mean", "attention", "first", "last")
        self.dropout = torch.nn.Dropout(dropout_p)
        if mode == "attention":
            self.key_proj = torch.nn.Linear(input_size, n_heads)
            self.value_proj = (
                torch.nn.Linear(input_size, input_size) if do_value_proj else None
            )
        self.output_size = input_size

    def forward(self, features, mask):
        device = features.device
        if self.mode == "attention" and isinstance(mask, tuple):
            position = torch.arange(features.shape[-2], device=device).reshape(
                [1] * (features.ndim - 2) + [features.shape[-2]]
            )
            mask = (mask[0].unsqueeze(-1) <= position) & (
                position < mask[1].unsqueeze(-1)
            )
            features = features.unsqueeze(-3)
        if isinstance(mask, tuple):
            original_dtype = features.dtype
            if features.dtype == torch.int or features.dtype == torch.long:
                features = features.float()
            begins, ends = mask
            if self.mode == "first":
                ends = torch.minimum(begins + 1, ends)
            if self.mode == "last":
                begins = torch.maximum(ends - 1, begins)
            begins = begins.expand(
                *features.shape[: begins.ndim - 1], begins.shape[-1]
            ).clamp_min(0)
            ends = ends.expand(
                *features.shape[: begins.ndim - 1], ends.shape[-1]
            ).clamp_min(0)
            final_shape = (*begins.shape, *features.shape[begins.ndim :])
            features = features.view(-1, features.shape[-2], features.shape[-1])
            begins = begins.reshape(
                features.shape[0],
                begins.numel() // features.shape[0] if len(features) else 0,
            )
            ends = ends.reshape(
                features.shape[0],
                ends.numel() // features.shape[0] if len(features) else 0,
            )

            max_window_size = (
                max(0, int((ends - begins).max())) if 0 not in ends.shape else 0
            )
            flat_indices = (
                torch.arange(max_window_size, device=device)[None, None, :]
                + begins[..., None]
            )
            flat_indices_mask = flat_indices < ends[..., None]
            flat_indices += (
                torch.arange(len(flat_indices), device=device)[:, None, None]
                * features.shape[1]
            )

            flat_indices = flat_indices[flat_indices_mask]
            res = F.embedding_bag(
                input=flat_indices,
                weight=self.dropout(features.reshape(-1, features.shape[-1])),
                offsets=torch.cat(
                    [
                        torch.tensor([0], device=device),
                        flat_indices_mask.sum(-1).reshape(-1),
                    ]
                )
                .cumsum(0)[:-1]
                .clamp_max(flat_indices.shape[0]),
                mode=self.mode if self.mode not in ("first", "last") else "max",
            ).reshape(final_shape)
            if res.dtype != original_dtype:
                res = res.type(original_dtype)
            return res
        elif torch.is_tensor(mask):
            features = features
            features = self.dropout(features)
            if self.mode == "first":
                mask = ~shift(mask.long(), n=1, dim=-1).cumsum(-1).bool() & mask
            elif self.mode == "last":
                mask = mask.flip(-1)
                mask = (~shift(mask.long(), n=1, dim=-1).cumsum(-1).bool() & mask).flip(
                    -1
                )

            if mask.ndim <= features.ndim - 1:
                mask = mask.unsqueeze(-1)
            if 0 in mask.shape:
                return features.sum(-2)
            if self.mode == "attention":
                weights = (
                    self.key_proj(features).masked_fill(~mask, -100000).softmax(-2)
                )  # ... tokens heads
                values = (
                    self.value_proj(features)
                    if self.value_proj is not None
                    else features
                )
                values = values.view(
                    *values.shape[:-1], weights.shape[-1], -1
                )  # ... tokens heads dim
                res = torch.einsum("...nhd,...nh->...hd", values, weights)
                return res.view(*res.shape[:-2], -1)
            elif self.mode == "max":
                features = (
                    features.masked_fill(~mask, -100000)
                    .max(-2)
                    .values.masked_fill(~(mask.any(-2)), 0)
                )
            elif self.mode == "abs-max":
                values, indices = features.abs().masked_fill(~mask, -100000).max(-2)
                features = features.gather(dim=-2, index=indices.unsqueeze(1)).squeeze(
                    1
                )
            elif self.mode in ("sum", "mean", "first", "last"):
                features = features.masked_fill(~mask, 0).sum(-2)
                if self.mode == "mean":
                    features = features / mask.float().sum(-2).clamp_min(1.0)
            elif self.mode == "softmax":
                weights = (
                    (features.detach() * self.alpha)
                    .masked_fill(~mask, -100000)
                    .softmax(-2)
                )
                features = torch.einsum(
                    "...nd,...nd->...d", weights, features.masked_fill(~mask, 0)
                )
            elif self.mode == "softmax-abs":
                weights = (
                    (features.detach().abs() * self.alpha)
                    .masked_fill(~mask, -100000)
                    .softmax(-2)
                )
                features = torch.einsum(
                    "...nd,...nd->...d", weights, features.masked_fill(~mask, 0)
                )
            return features


class ToyModel(nn.Module):
    def __init__(
        self,
        num_embedding=32006,
        embedding_dim=10,
        num_classes=2,
        dropout_pooler=0,
        pooler_mode="mean",
        dropout_classification_head=0.1,
        **kwargs,
    ):
        super().__init__()

        self.transformer = torch.nn.Embedding(
            num_embedding, embedding_dim=embedding_dim, padding_idx=1
        )

        # Pooling to focus on entities
        self.pooler = Pooler(
            mode=pooler_mode,
            dropout_p=dropout_pooler,  # or self.transformer.embeddings.dropout.p,
            input_size=embedding_dim,  # self.transformer.pooler.dense.out_features,
        )
        self.classification_head = SimpleClassificationHead(
            hidden_size=embedding_dim,
            num_classes=num_classes,
            dropout=dropout_classification_head,
        )

    def forward(self, input_ids, attention_mask, span_start, span_end, **kwargs):
        # last_hidden_state = self.transformer(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        # ).last_hidden_state
        last_hidden_state = self.transformer(input_ids)

        logits = self.classification_head(
            self.pooler(last_hidden_state, (span_start, span_end))
        ).squeeze()

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=-1)
        return logits, preds, probs


class QualifierModelv2(nn.Module):
    def __init__(
        self,
        path_embedding="/data/scratch/cse/camembert-EDS",
        num_classes=2,
        dropout_pooler=0,
        pooler_mode="mean",
        dropout_classification_head=0.1,
        **kwargs,
    ):
        super().__init__()

        # Embedding Layer
        self.transformer = AutoModel.from_pretrained(path_embedding)
        # Resize embedding
        if "tokenizer" in kwargs:
            # print("### tokenizer ###", len(kwargs["tokenizer"]))
            self.transformer.resize_token_embeddings(len(kwargs["tokenizer"]))

        # Pooling to focus on entities
        self.pooler = Pooler(
            mode=pooler_mode,
            dropout_p=dropout_pooler or self.transformer.embeddings.dropout.p,
            input_size=self.transformer.pooler.dense.out_features,
        )
        # self.classification_head = SimpleClassificationHead(
        #     hidden_size=embedding_dim,
        #     num_classes=num_classes,
        #     dropout=dropout_classification_head,
        # )

        self.classification_head = RobertaClassificationHead(
            hidden_size=self.transformer.pooler.dense.out_features,
            num_classes=num_classes,
            dropout=dropout_classification_head,
        )

    def forward(self, input_ids, attention_mask, span_start, span_end, **kwargs):
        last_hidden_state = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

        logits = self.classification_head(
            self.pooler(last_hidden_state, (span_start, span_end))
        ).squeeze(1)

        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        return logits, preds, probs
