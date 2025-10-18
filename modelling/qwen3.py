# https://github.com/huggingface/transformers/blob/v4.54.0/src/transformers/models/qwen3/modeling_qwen3.py
# NOTE: there are 2 key differences in numerics in this implementation
# - we use nn.RMSNorm, which does multiplication in FP32, while Qwen3RMSNorm does it in BF16.
# - we perform RoPE in FP32.

import torch
from torch import Tensor, nn
from transformers import Qwen3Config

from .attn import VarlenInfo, attention
from .rope import apply_rope, compute_rope
from .utils import load_hf_state_dict


class Qwen3Attention(nn.Module):
    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.head_dim = cfg.head_dim
        self.attention_dropout = cfg.attention_dropout
        qo_dim = cfg.num_attention_heads * cfg.head_dim
        kv_dim = cfg.num_key_value_heads * cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, qo_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, kv_dim, bias=False)
        self.o_proj = nn.Linear(qo_dim, cfg.hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = nn.RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps)

    def forward(self, x: Tensor, pos_embeds: Tensor, varlen_info: VarlenInfo | None = None) -> Tensor:
        hidden_shape = (*x.shape[:-1], -1, self.head_dim)
        q = apply_rope(self.q_norm(self.q_proj(x).view(hidden_shape)), pos_embeds)
        k = apply_rope(self.k_norm(self.k_proj(x).view(hidden_shape)), pos_embeds)
        v = self.v_proj(x).view(hidden_shape)

        dropout_p = self.attention_dropout if self.training else 0.0
        out = attention(q, k, v, varlen_info=varlen_info, is_causal=True, dropout_p=dropout_p)
        out = self.o_proj(out.flatten(-2))
        return out


class Qwen3MLP(nn.Module):
    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.input_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.self_attn = Qwen3Attention(cfg)
        self.post_attention_layernorm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = Qwen3MLP(cfg)

    def forward(self, x: Tensor, pos_embeds: Tensor, varlen_info: VarlenInfo | None = None) -> Tensor:
        x = x + self.self_attn(self.input_layernorm(x), pos_embeds, varlen_info)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3Model(nn.Module):
    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, cfg.pad_token_id)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(
        self,
        input_ids: Tensor | None = None,
        *,
        input_embeds: Tensor | None = None,
        pos_ids: Tensor | None = None,
        varlen_info: VarlenInfo | None = None,  # if this is present, pos_ids is redundant?
    ) -> Tensor:
        hidden_states = self.embed_tokens(input_ids) if input_embeds is None else input_embeds

        if pos_ids is None:
            pos_ids = torch.arange(hidden_states.shape[-2], device=hidden_states.device)
        pos_embeds = compute_rope(pos_ids, self.cfg.rope_theta, self.cfg.head_dim)
        # pos_embeds = pos_embeds.to(hidden_states.dtype)

        for layer in self.layers:
            hidden_states = layer(hidden_states, pos_embeds, varlen_info)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = Qwen3Model(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def forward(
        self,
        input_ids: Tensor | None = None,
        *,
        input_embeds: Tensor | None = None,
        pos_ids: Tensor | None = None,
        varlen_info: VarlenInfo | None = None,
    ) -> Tensor:
        hidden_states = self.model(
            input_ids,
            input_embeds=input_embeds,
            pos_ids=pos_ids,
            varlen_info=varlen_info,
        )
        logits = self.lm_head(hidden_states)
        return logits

    @staticmethod
    def from_pretrained(model_id: str) -> "Qwen3ForCausalLM":
        cfg = Qwen3Config.from_pretrained(model_id)
        with torch.device("meta"):
            model = Qwen3ForCausalLM(cfg)

        state_dict = load_hf_state_dict(model_id)
        model.load_state_dict(state_dict, assign=True)
        return model
