# https://github.com/OpenGVLab/InternVL/blob/eecca2aa/internvl_chat/internvl/model/internvl_chat

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import InternVLConfig, InternVLVisionConfig, Qwen3Config

from attn import attention
from qwen3 import Qwen3ForCausalLM
from utils import load_hf_state_dict

IMG_START_ID = 151669  # <img>
IMG_END_ID = 151670  # </img>
IMG_CONTEXT_ID = 151671  # <IMG_CONTEXT>


class InternAttention(nn.Module):
    def __init__(self, config: InternVLVisionConfig):
        super().__init__()
        embed_dim = config.hidden_size
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.num_heads = config.num_attention_heads
        self.dropout = config.attention_dropout

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(hidden_states).unflatten(2, (3, self.num_heads, -1)).unbind(2)
        dropout_p = self.dropout if self.training else 0.0
        out = attention(q, k, v, dropout_p=dropout_p)
        out = self.proj(out.flatten(2))
        return out


class InternMLP(nn.Sequential):
    def __init__(self, config: InternVLVisionConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)


class InternVisionEncoderLayer(nn.Module):
    def __init__(self, config: InternVLVisionConfig) -> None:
        super().__init__()
        self.attn = InternAttention(config)
        self.mlp = InternMLP(config)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.ls1 = nn.Parameter(torch.ones(config.hidden_size))
        self.ls2 = nn.Parameter(torch.ones(config.hidden_size))
        # NOTE: original code has DropPath

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x)) * self.ls1
        x = x + self.mlp(self.norm2(x)) * self.ls2
        return x


class InternVisionEmbeddings(nn.Module):
    def __init__(self, config: InternVLVisionConfig) -> None:
        super().__init__()
        self.nH = config.image_size[0] // config.patch_size[0]
        self.nW = config.image_size[1] // config.patch_size[1]
        num_positions = self.nH * self.nW + 1
        self.class_embedding = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.patch_embedding = nn.Conv2d(3, config.hidden_size, config.patch_size, config.patch_size)
        self.position_embedding = nn.Parameter(torch.randn(1, num_positions, config.hidden_size))

    def _get_pos_embed(self, H: int, W: int) -> Tensor:
        if 1 + H * W == self.position_embedding.shape[1]:
            return self.position_embedding

        cls_pos_embeds = self.position_embedding[:, :1]
        pos_embeds = self.position_embedding[:, 1:]

        pos_embeds = pos_embeds.transpose(1, 2).unflatten(2, (self.nH, self.nW))
        pos_embeds = F.interpolate(pos_embeds.float(), size=(H, W), mode="bicubic", align_corners=False)
        pos_embeds = pos_embeds.to(cls_pos_embeds.dtype)
        pos_embeds = pos_embeds.flatten(2).transpose(1, 2)

        return torch.cat([cls_pos_embeds, pos_embeds], dim=1)

    def forward(self, pixel_values: Tensor) -> Tensor:
        patch_embeds = self.patch_embedding(pixel_values)
        B, _, H, W = patch_embeds.shape

        # prepend [CLS] token
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(B, 1, -1).to(patch_embeds.dtype)
        embeds = torch.cat([class_embeds, patch_embeds], dim=1)

        # add learned position embeddings
        embeds = embeds + self._get_pos_embed(H, W)
        return embeds


class InternVisionEncoder(nn.Module):
    def __init__(self, config: InternVLVisionConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList([InternVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class InternVisionModel(nn.Module):
    def __init__(self, config: InternVLVisionConfig) -> None:
        super().__init__()
        self.embeddings = InternVisionEmbeddings(config)
        self.encoder = InternVisionEncoder(config)

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(self.embeddings(x))[:, 1:]  # exclude [CLS] token


def pixel_shuffle(x: Tensor, patch_size: int = 2):
    N, H, W, C = x.shape
    x = x.reshape(N, H // patch_size, patch_size, W // patch_size, patch_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5)
    x = x.reshape(N, H // patch_size, W // patch_size, -1)
    return x


class InternVLChatModel(nn.Module):
    def __init__(self, config: InternVLConfig):
        super().__init__()
        self.config = config
        self.downsample = int(1 / config.downsample_ratio)
        vit_dim = config.vision_config.hidden_size
        llm_dim = config.text_config.hidden_size

        self.vision_model = InternVisionModel(config.vision_config)
        self.language_model = Qwen3ForCausalLM(config.text_config)
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_dim * self.downsample * self.downsample),
            nn.Linear(vit_dim * self.downsample * self.downsample, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, pixel_values: Tensor, input_ids: Tensor, *, pos_ids: Tensor | None = None) -> Tensor:
        # vision encoder, pixel (un)shuffle, and MLP projection
        vit_embeds = self.vision_model(pixel_values)
        imgH, img_W = pixel_values.shape[2:]
        vit_patch_size = self.config.vision_config.patch_size  # assume HW order
        vit_embeds = vit_embeds.unflatten(1, (imgH // vit_patch_size[0], img_W // vit_patch_size[1]))
        vit_embeds = pixel_shuffle(vit_embeds, patch_size=self.downsample)
        vit_embeds = vit_embeds.flatten(1, 2)
        vit_embeds = self.mlp1(vit_embeds)

        input_embeds = self.language_model.model.embed_tokens(input_ids)

        # replace <IMG_CONTEXT> token with image embeddings
        input_embeds = input_embeds.clone()
        input_embeds[input_ids == IMG_CONTEXT_ID] = vit_embeds

        return self.language_model(input_embeds=input_embeds, pos_ids=pos_ids)

    @staticmethod
    def from_pretrained(model_id: str):
        cfg = InternVLConfig.from_pretrained(model_id)
        cfg.text_config = Qwen3Config(**cfg.llm_config)
        with torch.device("meta"):
            model = InternVLChatModel(cfg)

        state_dict = load_hf_state_dict(model_id)
        model.load_state_dict(state_dict, assign=True)
        return model
