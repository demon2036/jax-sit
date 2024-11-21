import math
import random
from typing import Optional, Callable

import numpy as np
import flax.linen as nn
import jax
import jax.numpy as jnp

use_fast_variance = False


class MLP(nn.Module):
    hidden_size: int
    projector_dim: int
    z_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.projector_dim, name='0')(x)
        x = nn.swish(x)
        x = nn.Dense(self.projector_dim, name='2')(x)
        x = nn.swish(x)
        x = nn.Dense(self.z_dim, name='4')(x)
        return x


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
    hidden_size: int
    frequency_embedding_size: int = 256
    max_period: int = 10000

    @nn.compact
    def __call__(self, t):
        t_freq = self.positional_embedding(t, dim=self.frequency_embedding_size, max_period=self.max_period)
        t_emb = nn.Dense(self.hidden_size, name='0')(t_freq)
        t_emb = nn.swish(t_emb)
        t_emb = nn.Dense(self.hidden_size, name='2')(t_emb)
        return t_emb

    @staticmethod
    def positional_embedding(t, dim, max_period=10000):
        half_dim = dim // 2
        freqs = jnp.exp(-math.log(max_period) * jnp.arange(half_dim) / half_dim)
        args = t[:, None] * freqs[None, :]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2:
            embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
        return embedding


class LabelEmbedder(nn.Module):
    num_classes: int
    hidden_size: int
    dropout_prob: float

    @nn.compact
    def __call__(self, labels, train: bool, force_drop_ids=None, rng=None):
        use_cfg_embedding = self.dropout_prob > 0
        # embedding_table = self.param(
        #     "embedding_table",
        #     nn.initializers.normal(stddev=0.02),
        #     (self.num_classes + int(use_cfg_embedding), self.hidden_size),
        # )


        if train and self.dropout_prob > 0 or force_drop_ids is not None:

            if rng is None:
                rng = self.make_rng("dropout")

            labels = self.token_drop(labels, force_drop_ids, rng)

        embeddings = nn.Embed(
            num_embeddings=self.num_classes + int(use_cfg_embedding), features=self.hidden_size, name='embedding'
        )(labels)

        return embeddings

    def token_drop(self, labels, force_drop_ids, rng):
        drop_ids = (
            force_drop_ids == 1
            if force_drop_ids is not None
            else jax.random.uniform(rng, shape=labels.shape) < self.dropout_prob
        )
        labels = jnp.where(drop_ids, self.num_classes, labels)
        return labels


class Mlp(nn.Module):
    """
    Flax implementation of the MLP as used in Vision Transformer, MLP-Mixer, etc.
    """
    in_features: int
    hidden_features: Optional[int] = None
    out_features: Optional[int] = None
    act_layer: Callable[..., nn.Module] = nn.gelu
    norm_layer: Optional[Callable[..., nn.Module]] = None
    bias: bool = True
    drop: float = 0.0
    use_conv: bool = False

    @nn.compact
    def __call__(self, x, det=True):
        out_features = self.out_features or self.in_features
        hidden_features = self.hidden_features or self.in_features

        # Choose the linear or convolutional layer
        linear_layer = nn.Dense

        # First layer
        x = linear_layer(features=hidden_features, use_bias=self.bias, name='fc1')(x)
        # x = self.act_layer(name='act')(x)
        x = self.act_layer(x, approximate=True)
        x = nn.Dropout(self.drop)(x, deterministic=det)

        # Optional normalization
        if self.norm_layer:
            x = self.norm_layer()(x)

        # Second layer
        x = linear_layer(features=out_features, use_bias=self.bias, name='fc2')(x)
        x = nn.Dropout(self.drop)(x, deterministic=det)

        return x


class Attention(nn.Module):
    num_heads: int
    dim: int
    qkv_bias: bool = False
    qk_norm: bool = False
    attn_drop: float = 0.
    proj_drop: float = 0.
    fused_attn: bool = False  # Flax doesn't support fused attention natively; assume False

    def setup(self):
        assert self.dim % self.num_heads == 0, 'dim should be divisible by num_heads'
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Dense(self.dim * 3, use_bias=self.qkv_bias)
        self.q_norm = nn.LayerNorm() if self.qk_norm else Identity()
        self.k_norm = nn.LayerNorm() if self.qk_norm else Identity()

        self.proj = nn.Dense(self.dim)

    def __call__(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)

        # Scaled Dot-Product Attention
        q = q * self.scale
        attn = jnp.matmul(q, k.transpose(0, 1, 3, 2))  # B, heads, N, N
        attn = nn.softmax(attn, axis=-1)
        x = jnp.matmul(attn, v)  # B, heads, N, head_dim
        x = x.transpose(0, 2, 1, 3).reshape(B, N, C)  # B, N, C (flatten heads and head_dim)

        x = self.proj(x)
        return x


class Identity(nn.Module):
    def __call__(self, x, det=True):
        return x


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding.
    """
    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    embed_dim: int = 768
    norm_layer: Optional[nn.Module] = None
    flatten: bool = True

    def setup(self):
        # self.patch_size = (self.patch_size, self.patch_size)  # Ensure tuple for both dimensions
        self.grid_size = (self.img_size // self.patch_size, self.img_size // self.patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv(
            features=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=self.patch_size,
            use_bias=True,
        )
        self.norm = self.norm_layer or Identity()

    def __call__(self, x):
        B, H, W, C = x.shape  # NHWC format
        assert H == self.img_size and W == self.img_size, (
            f"Input image size ({H}, {W}) doesn't match expected size {self.img_size}."
        )
        x = self.proj(x)  # Shape: [B, grid_size[0], grid_size[1], embed_dim]
        if self.flatten:
            x = x.reshape(B, -1, self.embed_dim)  # Flatten spatial dimensions to [B, num_patches, embed_dim]
        return self.norm(x)


#################################################################################
#                                 Core SiT Model                                #
#################################################################################
class SiTBlock(nn.Module):
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0
    qk_norm: bool = False
    fused_attn: bool = False

    @nn.compact
    def __call__(self, x, c):
        # LayerNorm without affine parameters
        norm1 = nn.LayerNorm(use_scale=False, use_bias=False, epsilon=1e-6, use_fast_variance=use_fast_variance)
        norm2 = nn.LayerNorm(use_scale=False, use_bias=False, epsilon=1e-6, use_fast_variance=use_fast_variance)

        # Attention and MLP
        attn = Attention(dim=self.hidden_size, num_heads=self.num_heads, qkv_bias=True, qk_norm=self.qk_norm,
                         name='attn')
        if self.fused_attn:
            attn.fused_attn = self.fused_attn

        mlp_hidden_dim = int(self.hidden_size * self.mlp_ratio)
        mlp = Mlp(
            in_features=self.hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=nn.gelu,
            drop=0.0, name='mlp'
        )

        # AdaLN Modulation
        adaLN_modulation = nn.Sequential([
            nn.swish,
            nn.Dense(6 * self.hidden_size),
        ])
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(adaLN_modulation(c), 6, axis=-1)

        # Adaptive LayerNorm and updates
        x = x + gate_msa[..., None, :] * attn(modulate(norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp[..., None, :] * mlp(modulate(norm2(x), shift_mlp, scale_mlp))

        return x


class FinalLayer(nn.Module):
    hidden_size: int
    patch_size: int
    out_channels: int

    @nn.compact
    def __call__(self, x, c):
        # Layer Normalization without affine parameters
        x = nn.LayerNorm(use_scale=False, use_bias=False, epsilon=1e-6)(x)

        # AdaLN modulation
        adaLN = nn.Sequential([
            nn.swish,
            nn.Dense(2 * self.hidden_size, name='ada_dense'),
        ])
        shift, scale = jnp.split(adaLN(c), 2, axis=-1)

        # Modulation
        x = x * (1 + scale[..., None, :]) + shift[..., None, :]

        # Linear transformation
        x = nn.Dense(self.patch_size * self.patch_size * self.out_channels, name='proj')(x)
        return x


class SiT(nn.Module):
    path_type: str = "edm"
    input_size: int = 32
    patch_size: int = 2
    in_channels: int = 4
    out_channels: int = 4
    hidden_size: int = 1152
    decoder_hidden_size: int = 1152
    encoder_depth: int = 8
    depth: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    class_dropout_prob: float = 0.1
    num_classes: int = 1000
    use_cfg: bool = False
    z_dims: int = 768
    projector_dim: int = 2048
    block_kwargs: dict = None  # e.g., fused_attn, qk_norm

    def setup(self):
        self.x_embedder = PatchEmbed(
            img_size=self.input_size,
            patch_size=self.patch_size,
            # in_channels=self.in_channels,
            embed_dim=self.hidden_size,
            # bias=True,
        )

        self.t_embedder = TimestepEmbedder(hidden_size=self.hidden_size)
        self.y_embedder = LabelEmbedder(
            num_classes=self.num_classes, hidden_size=self.hidden_size, dropout_prob=self.class_dropout_prob
        )
        self.pos_embed = self.param(
            "pos_embed",
            nn.initializers.zeros,
            (1, self.x_embedder.num_patches, self.hidden_size),
        )

        self.blocks = [
            SiTBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                **(self.block_kwargs or {})
            )
            for _ in range(self.depth)
        ]

        # self.projectors = [
        #     Mlp(hidden_size=self.hidden_size, hidden_features=self.projector_dim, out_features=self.z_dims)
        #     for z_dim in self.z_dims
        # ]

        self.final_layer = FinalLayer(
            hidden_size=self.decoder_hidden_size,
            patch_size=self.patch_size,
            out_channels=self.in_channels,
        )

    def unpatchify(self, x, patch_size=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape((x.shape[0], h, w, p, p, c))
        x = jnp.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape((x.shape[0], c, h * p, h * p))
        return imgs


    @nn.compact
    def __call__(self, x, t, y, return_logvar=False):
        x = self.x_embedder(x) + self.pos_embed
        N, T, D = x.shape

        t_embed = self.t_embedder(t)
        y_embed = self.y_embedder(y, train=False)#self.is_mutable_collection("params")

        c = t_embed + y_embed

        zs = []
        for i, block in enumerate(self.blocks):
            x = block(x, c)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x

        return x, zs


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb



def SiT_XL_2(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=2, num_heads=16, **kwargs)