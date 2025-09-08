# https://github.com/huggingface/transformers/blob/v4.54.0/src/transformers/models/qwen3/modeling_qwen3.py
# NOTE: there are 2 key differences in numerics in this implementation
# - we use nn.RMSNorm, which does multiplication in FP32, while Qwen3RMSNorm does it in BF16.
# - we perform RoPE in FP32.

import torch
from torch import Tensor, nn
from transformers import Qwen3Config

from .attn import attention
from .utils import load_hf_state_dict


# NOTE: if x and pos_embeds are BF16, the computation is done in BF16
def apply_rope(x: Tensor, pos_embeds: Tensor) -> Tensor:
    # x: [*, L, num_heads, dim]
    # pos_embeds: [*, L, dim]
    # pos_embeds may have fewer leading dimensions than x's
    x1, x2 = x.chunk(2, dim=-1)
    cos, sin = pos_embeds.unsqueeze(-2).chunk(2, dim=-1)

    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.cat([o1, o2], dim=-1).to(x.dtype)


class Qwen3Attention(nn.Module):
    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.head_dim = cfg.head_dim
        self.attention_dropout = cfg.attention_dropout
        self.q_proj = nn.Linear(
            cfg.hidden_size,
            cfg.num_attention_heads * self.head_dim,
            bias=cfg.attention_bias,
        )
        self.k_proj = nn.Linear(
            cfg.hidden_size,
            cfg.num_key_value_heads * self.head_dim,
            bias=cfg.attention_bias,
        )
        self.v_proj = nn.Linear(
            cfg.hidden_size,
            cfg.num_key_value_heads * self.head_dim,
            bias=cfg.attention_bias,
        )
        self.o_proj = nn.Linear(
            cfg.num_attention_heads * self.head_dim,
            cfg.hidden_size,
            bias=cfg.attention_bias,
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)

    def forward(self, x: Tensor, pos_embeds: Tensor) -> Tensor:
        hidden_shape = (*x.shape[:-1], -1, self.head_dim)
        q = apply_rope(self.q_norm(self.q_proj(x).view(hidden_shape)), pos_embeds)
        k = apply_rope(self.k_norm(self.k_proj(x).view(hidden_shape)), pos_embeds)
        v = self.v_proj(x).view(hidden_shape)

        dropout_p = self.attention_dropout if self.training else 0.0
        out = attention(q, k, v, is_causal=True, dropout_p=dropout_p)
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

    def forward(self, x: Tensor, pos_embeds: Tensor) -> Tensor:
        x = x + self.self_attn(self.input_layernorm(x), pos_embeds)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


def compute_rope(pos_ids: Tensor, rope_theta: float, dim: int) -> Tensor:
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float, device=pos_ids.device) / dim))
    freqs = pos_ids.unsqueeze(-1) * inv_freq
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)  # [*pos_ids.shape, dim]


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
    ) -> Tensor:
        hidden_states = self.embed_tokens(input_ids) if input_embeds is None else input_embeds

        if pos_ids is None:
            pos_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device)
        pos_embeds = compute_rope(pos_ids, self.cfg.rope_theta, self.cfg.head_dim)
        # pos_embeds = pos_embeds.to(hidden_states.dtype)

        for layer in self.layers:
            hidden_states = layer(hidden_states, pos_embeds)
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
    ) -> Tensor:
        hidden_states = self.model(input_ids, input_embeds=input_embeds, pos_ids=pos_ids)
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
