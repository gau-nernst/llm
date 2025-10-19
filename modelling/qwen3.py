# https://github.com/huggingface/transformers/blob/v4.54.0/src/transformers/models/qwen3/modeling_qwen3.py
# NOTE: there are 3 key differences in numerics in this implementation
# - we use nn.RMSNorm, which does multiplication in FP32, while Qwen3RMSNorm does it in BF16.
# - we perform RoPE in FP32.
# - qkv and gate/up are merged.
# init weights logic follows torchtitan

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint
from transformers import Qwen3Config

from .attn import VarlenInfo, attention
from .rope import apply_rope, compute_rope
from .utils import Linear, RMSNorm, load_hf_state_dict, make_merge_hook


class Qwen3Attention(nn.Module):
    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.head_dim = cfg.head_dim
        self.attention_dropout = cfg.attention_dropout
        self.num_qo_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        qo_dim = cfg.num_attention_heads * cfg.head_dim
        kv_dim = cfg.num_key_value_heads * cfg.head_dim

        self.qkv_proj = Linear(cfg.hidden_size, qo_dim + kv_dim * 2, bias=False)
        self.o_proj = Linear(qo_dim, cfg.hidden_size, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = RMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps)
        self.register_load_state_dict_pre_hook(make_merge_hook(["q", "k", "v"], "qkv"))

    def init_weights(self, std: float, rng: torch.Generator | None = None):
        nn.init.trunc_normal_(self.qkv_proj.weight, 0.0, 0.02, generator=rng)
        nn.init.trunc_normal_(self.o_proj.weight, 0.0, std, generator=rng)
        self.q_norm.reset_parameters()
        self.k_norm.reset_parameters()

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
        self.gate_up_proj = Linear(cfg.hidden_size, cfg.intermediate_size * 2, bias=False)
        self.down_proj = Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)
        self.act_fn = nn.SiLU()
        self.register_load_state_dict_pre_hook(make_merge_hook(["gate", "up"], "gate_up"))

    def init_weights(self, std: float, rng: torch.Generator | None = None):
        gate, up = self.gate_up_proj.weight.chunk(2, dim=0)
        nn.init.trunc_normal_(gate, 0.0, 0.02, generator=rng)
        nn.init.trunc_normal_(up, 0.0, std, generator=rng)
        nn.init.trunc_normal_(self.down_proj.weight, 0.0, std, generator=rng)

    def forward(self, x: Tensor) -> Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen3Config, layer_id: int) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.self_attn = Qwen3Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.mlp = Qwen3MLP(cfg)
        self.layer_id = layer_id

    def init_weights(self, rng: torch.Generator | None = None):
        std = 0.02 / (2 * (self.layer_id + 1)) ** 0.5
        self.input_layernorm.reset_parameters()
        self.self_attn.init_weights(std, rng)
        self.post_attention_layernorm.reset_parameters()
        self.mlp.init_weights(std, rng)

    def forward(self, x: Tensor, pos_embeds: Tensor, varlen_info: VarlenInfo | None = None) -> Tensor:
        x = x + self.self_attn(self.input_layernorm(x), pos_embeds, varlen_info)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3Model(nn.Module):
    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.compute_dtype: torch.dtype | None = None
        self.act_ckpt = False
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size, cfg.pad_token_id)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def init_weights(self, rng: torch.Generator | None = None):
        nn.init.normal_(self.embed_tokens.weight, generator=rng)
        for layer in self.layers:
            layer.init_weights(rng)
        self.norm.reset_parameters()

    def forward(
        self,
        input_ids: Tensor | None = None,
        *,
        input_embeds: Tensor | None = None,
        pos_ids: Tensor | None = None,
        varlen_info: VarlenInfo | None = None,  # if this is present, pos_ids is redundant?
    ) -> Tensor:
        hidden_states = self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        hidden_states = hidden_states.to(self.compute_dtype)

        if pos_ids is None:
            pos_ids = torch.arange(hidden_states.shape[-2], device=hidden_states.device)
        pos_embeds = compute_rope(pos_ids, self.cfg.rope_theta, self.cfg.head_dim)
        # pos_embeds = pos_embeds.to(hidden_states.dtype)

        for layer in self.layers:
            if self.act_ckpt:
                hidden_states = checkpoint(layer, hidden_states, pos_embeds, varlen_info, use_reentrant=False)
            else:
                hidden_states = layer(hidden_states, pos_embeds, varlen_info)
        hidden_states = self.norm(hidden_states)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = Qwen3Model(cfg)
        self.lm_head = Linear(cfg.hidden_size, cfg.vocab_size, bias=False) if not cfg.tie_word_embeddings else None

    def init_weights(self, rng: torch.Generator | None = None):
        self.model.init_weights(rng)
        if self.lm_head is not None:
            std = self.cfg.hidden_size**-0.5
            cutoff = 3 * std
            nn.init.trunc_normal_(self.lm_head.weight, 0, std, -cutoff, cutoff, generator=rng)

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
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            w = self.model.embed_tokens.weight.to(hidden_states.dtype)
            logits = F.linear(hidden_states, w)
        return logits

    @staticmethod
    def from_pretrained(model_id: str) -> "Qwen3ForCausalLM":
        cfg = Qwen3Config.from_pretrained(model_id)
        with torch.device("meta"):
            model = Qwen3ForCausalLM(cfg)

        state_dict = load_hf_state_dict(model_id)
        if cfg.tie_word_embeddings and "lm_head.weight" in state_dict:
            state_dict.pop("lm_head.weight")
        model.load_state_dict(state_dict, assign=True)
        return model
