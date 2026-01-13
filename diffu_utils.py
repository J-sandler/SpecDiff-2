#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional

import torch
from transformers.cache_utils import DynamicCache


class DreamKVCacheWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs,
    ):
        if use_cache is None and hasattr(self.model, "config"):
            use_cache = bool(getattr(self.model.config, "use_cache", False))
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        if return_dict is None and hasattr(self.model, "config"):
            return_dict = bool(getattr(self.model.config, "use_return_dict", True))

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            num_logits_to_keep=num_logits_to_keep,
            **kwargs,
        )
        if return_dict:
            has_past = hasattr(outputs, "past_key_values")
            if not has_past:
                outputs.past_key_values = past_key_values
                try:
                    outputs["past_key_values"] = past_key_values
                except Exception:
                    pass
            return outputs
        return outputs


def ensure_kv_cache_model(model: torch.nn.Module) -> torch.nn.Module:
    model_name = model.__class__.__name__
    if model_name == "DreamModel":
        return DreamKVCacheWrapper(model)
    return model
