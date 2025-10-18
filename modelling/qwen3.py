# https://github.com/huggingface/transformers/blob/v4.54.0/src/transformers/models/qwen3/modeling_qwen3.py
# NOTE: there are 3 key differences in numerics in this implementation
# - we use nn.RMSNorm, which does multiplication in FP32, while Qwen3RMSNorm does it in BF16.
# - we perform RoPE in FP32.
# - qkv and gate/up are merged.

import torch
from torch import Tensor, nn
from transformers import Qwen3Config

from .attn import VarlenInfo, attention
from .rope import apply_rope, compute_rope
from .utils import load_hf_state_dict


def make_merge_hook(old_keys: list[str], new_key: str):
    def hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if f"{prefix}{old_keys[0]}_proj.weight" not in state_dict:
            return
        w_list = [state_dict.pop(f"{prefix}{key}_proj.weight") for key in old_keys]
        state_dict[f"{prefix}{new_key}_proj.weight"] = torch.cat(w_list, dim=0)

    return hook


class Qwen3Attention(nn.Module):
    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.head_dim = cfg.head_dim
        self.attention_dropout = cfg.attention_dropout
        self.num_qo_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        qo_dim = cfg.num_attention_heads * cfg.head_dim
        kv_dim = cfg.num_key_value_heads * cfg.head_dim

        self.qkv_proj = nn.Linear(cfg.hidden_size, qo_dim + kv_dim * 2, bias=False)
        self.o_proj = nn.Linear(qo_dim, cfg.hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = nn.RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps)
        self.register_load_state_dict_pre_hook(make_merge_hook(["q", "k", "v"], "qkv"))

    def forward(self, x: Tensor, pos_embeds: Tensor, varlen_info: VarlenInfo | None = None) -> Tensor:
        qkv = self.qkv_proj(x).view(*x.shape[:-1], -1, self.head_dim)
        q, k, v = qkv.split((self.num_qo_heads, self.num_kv_heads, self.num_kv_heads), dim=-2)
        q = apply_rope(self.q_norm(q), pos_embeds)
        k = apply_rope(self.k_norm(k), pos_embeds)

        dropout_p = self.attention_dropout if self.training else 0.0
        out = attention(q, k, v, varlen_info=varlen_info, is_causal=True, dropout_p=dropout_p)
        out = self.o_proj(out.flatten(-2))
        return out


class Qwen3MLP(nn.Module):
    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        self.register_load_state_dict_pre_hook(make_merge_hook(["gate", "up"], "gate_up"))

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)


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
