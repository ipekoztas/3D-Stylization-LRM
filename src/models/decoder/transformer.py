# Copyright (c) 2023, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn


class BasicTransformerBlock(nn.Module):
    """
    Transformer block that takes in a cross-attention condition and another modulation vector applied to sub-blocks.
    """
    # use attention from torch.nn.MultiHeadAttention
    # Block contains a cross-attention layer, a self-attention layer, and a MLP
    def __init__(
        self, 
        inner_dim: int, 
        cond_dim: int, 
        num_heads: int, 
        eps: float,
        attn_drop: float = 0., 
        attn_bias: bool = False,
        mlp_ratio: float = 4., 
        mlp_drop: float = 0.,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(inner_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=inner_dim, num_heads=num_heads, kdim=cond_dim, vdim=cond_dim,
            dropout=attn_drop, bias=attn_bias, batch_first=True)
        self.norm2 = nn.LayerNorm(inner_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=inner_dim, num_heads=num_heads,
            dropout=attn_drop, bias=attn_bias, batch_first=True)
        self.norm3 = nn.LayerNorm(inner_dim)
        self.mlp = nn.Sequential(
            nn.Linear(inner_dim, int(inner_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(mlp_drop),
            nn.Linear(int(inner_dim * mlp_ratio), inner_dim),
            nn.Dropout(mlp_drop),
        )

    def forward(self, x, cond, i, alpha, content_layers):
        # x: [N, L, D] or [x1, x2]
        # cond: [content_feats] or [content_feats, style_feats]
        if len(cond) == 2:
            # Style injection mode
            x1, x2 = x[0], x[1]
            content, style = cond[0], cond[1]
            if i <= content_layers:
                x1 = x1 + self.cross_attn(self.norm1(x1), content, content)[0]
            else:
                x1 = x1 + (1-alpha)*self.cross_attn(self.norm1(x1), content, content)[0] + (alpha)*self.cross_attn(self.norm1(x1), style, style)[0]
            x2 = x2 + self.cross_attn(self.norm1(x2), style, style)[0]

            before_sa1 = self.norm2(x1)
            before_sa2 = self.norm2(x2)
            x1 = x1 + self.self_attn(before_sa1, before_sa1, before_sa1)[0]
            x2 = x2 + self.self_attn(before_sa2, before_sa2, before_sa2)[0]

            x1 = x1 + self.mlp(self.norm3(x1))
            x2 = x2 + self.mlp(self.norm3(x2))

            return [x1, x2]
        else:
            # No style, only content
            x1 = x[0] if isinstance(x, list) else x
            content = cond[0]
            x1 = x1 + self.cross_attn(self.norm1(x1), content, content)[0]
            before_sa1 = self.norm2(x1)
            x1 = x1 + self.self_attn(before_sa1, before_sa1, before_sa1)[0]
            x1 = x1 + self.mlp(self.norm3(x1))
            return [x1]


class TriplaneTransformer(nn.Module):
    """
    Transformer with condition that generates a triplane representation.
    
    Reference:
    Timm: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L486
    """
    def __init__(
        self, 
        inner_dim: int, 
        image_feat_dim: int,
        triplane_low_res: int, 
        triplane_high_res: int, 
        triplane_dim: int,
        num_layers: int, 
        num_heads: int,
        eps: float = 1e-6,
    ):
        super().__init__()

        # attributes
        self.triplane_low_res = triplane_low_res
        self.triplane_high_res = triplane_high_res
        self.triplane_dim = triplane_dim

        # modules
        # initialize pos_embed with 1/sqrt(dim) * N(0, 1)
        self.pos_embed = nn.Parameter(torch.randn(1, 3*triplane_low_res**2, inner_dim) * (1. / inner_dim) ** 0.5)
        self.layers = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim=inner_dim, cond_dim=image_feat_dim, num_heads=num_heads, eps=eps)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(inner_dim, eps=eps)
        self.deconv = nn.ConvTranspose2d(inner_dim, triplane_dim, kernel_size=2, stride=2, padding=0)
        self.num_layers = num_layers

    def forward(self, image_feats, alpha, style_layers):
        # image_feats: [content_feats] or [content_feats, style_feats]
        N = image_feats[0].shape[0]
        H = W = self.triplane_low_res
        L = 3 * H * W
        content_layers = self.num_layers - style_layers
        x = self.pos_embed.repeat(N, 1, 1)  # [N, L, D]
        i = 1
        if len(image_feats) == 2:
            # Style injection mode
            for layer in self.layers:
                if i == 1:
                    x = layer([x, x], image_feats, i, alpha, content_layers)
                else:
                    x = layer(x, image_feats, i, alpha, content_layers)
                i += 1
            x = self.norm(x[0])
        else:
            # No style, only content
            for layer in self.layers:
                if i == 1:
                    x = layer([x], image_feats, i, alpha, content_layers)
                else:
                    x = layer(x, image_feats, i, alpha, content_layers)
                i += 1
            x = self.norm(x[0])

        # separate each plane and apply deconv
        x = x.view(N, 3, H, W, -1)
        x = torch.einsum('nihwd->indhw', x)  # [3, N, D, H, W]
        x = x.contiguous().view(3*N, -1, H, W)  # [3*N, D, H, W]
        x = self.deconv(x)  # [3*N, D', H', W']
        x = x.view(3, N, *x.shape[-3:])  # [3, N, D', H', W']
        x = torch.einsum('indhw->nidhw', x)  # [N, 3, D', H', W']
        x = x.contiguous()

        return x
