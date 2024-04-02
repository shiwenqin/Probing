"""This rewrite is a simplified version of the proposed changes that actually compiles statically in torch 2.0.

This model is the final, optimized crammed model.

Not all ablations discussed in the paper are implemented as switches in this version,
for all those, check scriptable_bert.py on the old branch.

"""
import torch
from transformers import PretrainedConfig, PreTrainedModel

from typing import Optional
from omegaconf import OmegaConf
from pathlib import Path

from .components import (
    _get_norm_fn,
    _get_nonlin_fn,
    EmbeddingComponent,
    PoolingComponent,
    PredictionHeadComponent,
    AttentionComponent,
    FFNComponent,
    get_extended_attention_mask,
    _init_module,
)

class crammedBertConfig(PretrainedConfig):
    model_type = "crammedBERT"

    def __init__(self, cfg_arch_container: dict = {}, **kwargs):
        self.arch = cfg_arch_container
        super().__init__(**kwargs)


def construct_crammed_bert(cfg_arch, vocab_size, downstream_classes):
    """See the config file for details on what is possible."""
    config = crammedBertConfig(OmegaConf.to_container(cfg_arch, resolve=True))
    config.arch["embedding"]["vocab_size"] = vocab_size
    config.arch["num_labels"] = downstream_classes

    model = ScriptableLM(config)
    return model


class TransformerLayer(torch.nn.Module):
    """A transformer-encoder structure based on the components from above."""

    def __init__(self, idx, cfg_arch):
        super().__init__()
        self.dropout = torch.nn.Dropout(cfg_arch.hidden_dropout_prob, inplace=False)
        self.norm1 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)
        self.norm2 = _get_norm_fn(cfg_arch.norm)(cfg_arch.hidden_size, eps=cfg_arch.norm_eps)
        self.attn = AttentionComponent(
            idx,
            cfg_arch.hidden_size,
            cfg_arch.attention,
            cfg_arch.use_bias,
        )
        self.LAYOUT = self.attn.LAYOUT

        self.ffn = FFNComponent(
            cfg_arch.hidden_size,
            cfg_arch.intermed_size,
            _get_nonlin_fn(cfg_arch.nonlin),
            cfg_arch.use_bias,
        )

    def forward(self, states, attention_mask: Optional[torch.Tensor] = None):
        states = states + self.dropout(self.attn(self.norm1(states), attention_mask))
        states = states + self.dropout(self.ffn(self.norm2(states)))
        return states


class ScriptableLM(PreTrainedModel):
    """Simplified transformer wrapper."""

    config_class = crammedBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.cfg = OmegaConf.create(config.arch)

        self.embedding = EmbeddingComponent(self.cfg.embedding, self.cfg.norm, self.cfg.norm_eps)
        self.layers = torch.nn.ModuleList([TransformerLayer(idx, self.cfg) for idx in range(self.cfg.num_transformer_layers)])
        self.seq_first = self.layers[0].LAYOUT == "[S B H]" if len(self.layers) > 0 else False
        self.use_causal_attention = self.cfg.attention.causal_attention

        if self.cfg.final_norm:
            self.final_norm = _get_norm_fn(self.cfg.norm)(self.cfg.hidden_size, eps=self.cfg.norm_eps)
        else:
            self.final_norm = torch.nn.Identity()

    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None):
        if attention_mask is not None:
            attention_mask = get_extended_attention_mask(attention_mask, input_ids.shape, self.use_causal_attention)
        hidden_states = self.embedding(input_ids)

        if self.seq_first:
            hidden_states = hidden_states.transpose(0, 1).contiguous()

        for i, layer_module in enumerate(self.layers):
            hidden_states = layer_module(hidden_states, attention_mask)

        if self.seq_first:
            hidden_states = hidden_states.transpose(0, 1).contiguous()

        return self.final_norm(hidden_states)
