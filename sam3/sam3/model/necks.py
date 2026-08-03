# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""Necks are the interface between a vision backbone and the rest of the detection model"""

from copy import deepcopy
from typing import List, Optional, Tuple

import torch

import torch.nn as nn
import torch.nn.functional as F


class TextScaleAdapter(nn.Module):
    """Inject text guidance into multi-level ViT features."""

    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        rank: int = 64,
        num_heads: int = 4,
        mixer_kernels=(1, 3, 5),
    ):
        super().__init__()
        if rank % num_heads != 0:
            raise ValueError(
                f"TextScale rank must be divisible by num_heads: "
                f"rank={rank}, num_heads={num_heads}."
            )

        self.rank = rank
        self.visual_down_adapter = nn.Conv2d(visual_dim, rank, kernel_size=1)
        self.text_proj_adapter = nn.Linear(text_dim, rank)

        self.spatial_mixer_convs = nn.ModuleList()
        in_channels = rank
        for kernel in mixer_kernels:
            if kernel % 2 != 1:
                raise ValueError(f"TextScale mixer kernels must be odd, got {kernel}.")
            self.spatial_mixer_convs.append(
                nn.Conv2d(
                    in_channels,
                    rank,
                    kernel_size=kernel,
                    padding=kernel // 2,
                )
            )
            in_channels += rank
        self.spatial_mixer_fuse = nn.Conv2d(
            rank * len(mixer_kernels), rank, kernel_size=1
        )
        self.cross_attn_adapter = nn.MultiheadAttention(
            embed_dim=rank,
            num_heads=num_heads,
            batch_first=True,
        )
        self.visual_up_adapter = nn.Conv2d(rank, visual_dim, kernel_size=1)
        nn.init.zeros_(self.visual_up_adapter.weight)
        nn.init.zeros_(self.visual_up_adapter.bias)

    def _mix_visual_context(self, visual_low: torch.Tensor) -> torch.Tensor:
        dense_inputs = [visual_low]
        dense_outputs = []
        for conv in self.spatial_mixer_convs:
            dense_out = F.gelu(conv(torch.cat(dense_inputs, dim=1)))
            dense_inputs.append(dense_out)
            dense_outputs.append(dense_out)
        return visual_low + self.spatial_mixer_fuse(torch.cat(dense_outputs, dim=1))

    def _to_batch_first_text(
        self,
        text_features: torch.Tensor,
        text_mask: Optional[torch.Tensor],
        batch_size: int,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if text_features.dim() != 3:
            raise ValueError(
                "TextScale expects text_features with shape [T, B, C] "
                f"or [B, T, C], got {tuple(text_features.shape)}."
            )
        if text_features.shape[1] == batch_size:
            text_features = text_features.permute(1, 0, 2)
        elif text_features.shape[0] != batch_size:
            raise ValueError(
                "Cannot infer the TextScale text-feature batch dimension: "
                f"text_features={tuple(text_features.shape)}, batch={batch_size}."
            )

        if text_mask is not None:
            if text_mask.dim() != 2:
                raise ValueError(
                    "TextScale expects text_mask with shape [B, T], "
                    f"got {tuple(text_mask.shape)}."
                )
            if text_mask.shape[0] != batch_size and text_mask.shape[1] == batch_size:
                text_mask = text_mask.transpose(0, 1)
            if text_mask.shape[0] != batch_size:
                raise ValueError(
                    "Cannot infer the TextScale text-mask batch dimension: "
                    f"text_mask={tuple(text_mask.shape)}, batch={batch_size}."
                )
            text_mask = text_mask.to(dtype=torch.bool, device=text_features.device)

        return text_features, text_mask

    def forward(
        self,
        visual: torch.Tensor,
        text_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, _, height, width = visual.shape
        text_features, text_mask = self._to_batch_first_text(
            text_features, text_mask, batch_size
        )

        visual_low = self.visual_down_adapter(visual)
        visual_dense = self._mix_visual_context(visual_low)

        visual_tokens = visual_dense.flatten(2).transpose(1, 2)
        text_tokens = self.text_proj_adapter(text_features)
        aligned_tokens, _ = self.cross_attn_adapter(
            query=visual_tokens,
            key=text_tokens,
            value=text_tokens,
            key_padding_mask=text_mask,
            need_weights=False,
        )
        aligned_tokens = visual_tokens + aligned_tokens
        aligned = aligned_tokens.transpose(1, 2).reshape(
            batch_size, self.rank, height, width
        )
        residual = self.visual_up_adapter(aligned)

        return residual


class Sam3DualViTDetNeck(nn.Module):
    def __init__(
        self,
        trunk: nn.Module,
        position_encoding: nn.Module,
        d_model: int,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        add_sam2_neck: bool = False,
        textscale_config: Optional[dict] = None,
    ):
        """
        SimpleFPN neck a la ViTDet
        (From detectron2, very lightly adapted)
        It supports a "dual neck" setting, where we have two identical necks (for SAM3 and SAM2), with different weights

        :param trunk: the backbone
        :param position_encoding: the positional encoding to use
        :param d_model: the dimension of the model
        """
        super().__init__()
        self.trunk = trunk
        self.position_encoding = position_encoding
        self.convs = nn.ModuleList()

        self.scale_factors = scale_factors
        self.use_textscale = bool(
            textscale_config and textscale_config.get("enabled", False)
        )
        self.textscale_visual_source = (
            textscale_config.get("visual_source", "multi_level")
            if self.use_textscale
            else "none"
        )
        valid_visual_sources = (
            {"multi_level", "shared_base"}
            if self.use_textscale
            else {"none"}
        )
        if self.textscale_visual_source not in valid_visual_sources:
            raise ValueError(
                "Unsupported TextScale visual_source: "
                f"{self.textscale_visual_source}; expected one of "
                f"{sorted(valid_visual_sources)}."
            )
        self.requires_text_features = self.use_textscale
        use_bias = True
        dim: int = self.trunk.channel_list[-1]

        self.textscale_adapters = None
        if self.use_textscale:
            rank = textscale_config.get("rank", 64)
            num_heads = textscale_config.get("num_heads", 4)
            mixer_kernels = tuple(
                textscale_config.get("mixer_kernels", (1, 3, 5))
            )
            self.textscale_adapters = nn.ModuleList(
                TextScaleAdapter(
                    visual_dim=dim,
                    text_dim=d_model,
                    rank=rank,
                    num_heads=num_heads,
                    mixer_kernels=mixer_kernels,
                )
                for _ in scale_factors
            )

        for _, scale in enumerate(scale_factors):
            current = nn.Sequential()

            if scale == 4.0:
                current.add_module(
                    "dconv_2x2_0",
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                )
                current.add_module(
                    "gelu",
                    nn.GELU(),
                )
                current.add_module(
                    "dconv_2x2_1",
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                )
                out_dim = dim // 4
            elif scale == 2.0:
                current.add_module(
                    "dconv_2x2",
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                )
                out_dim = dim // 2
            elif scale == 1.0:
                out_dim = dim
            elif scale == 0.5:
                current.add_module(
                    "maxpool_2x2",
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
                out_dim = dim
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            current.add_module(
                "conv_1x1",
                nn.Conv2d(
                    in_channels=out_dim,
                    out_channels=d_model,
                    kernel_size=1,
                    bias=use_bias,
                ),
            )
            current.add_module(
                "conv_3x3",
                nn.Conv2d(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                ),
            )
            self.convs.append(current)

        self.sam2_convs = None
        if add_sam2_neck:
            # Assumes sam2 neck is just a clone of the original neck
            self.sam2_convs = deepcopy(self.convs)

    def _mix_trunk_features(
        self,
        xs: List[torch.Tensor],
        text_features: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        base = xs[-1]
        if not self.use_textscale:
            return [base] * len(self.convs)

        if len(xs) != len(self.textscale_adapters):
            raise ValueError(
                "TextScale expects one trunk feature per FPN scale: "
                f"got {len(xs)} features and {len(self.textscale_adapters)} adapters."
            )

        adapter_inputs = xs
        if self.textscale_visual_source == "shared_base":
            adapter_inputs = [base] * len(self.textscale_adapters)

        if text_features is None:
            raise ValueError("TextScale requires text_features during image forward.")

        mixed = []
        for x, adapter in zip(adapter_inputs, self.textscale_adapters):
            residual = adapter(x, text_features=text_features, text_mask=text_mask)
            if residual.shape[-2:] != base.shape[-2:]:
                residual = F.interpolate(
                    residual,
                    size=base.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            mixed.append(base + residual)
        return mixed

    def forward(
        self,
        tensor_list: List[torch.Tensor],
        text_features: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        Optional[List[torch.Tensor]],
        Optional[List[torch.Tensor]],
    ]:
        xs = self.trunk(tensor_list)
        sam3_out, sam3_pos = [], []
        sam2_out, sam2_pos = None, None
        if self.sam2_convs is not None:
            sam2_out, sam2_pos = [], []
        xs_mixed = self._mix_trunk_features(
            xs, text_features=text_features, text_mask=text_mask
        )
        for i in range(len(self.convs)):
            x = xs_mixed[i]
            sam3_x_out = self.convs[i](x)
            sam3_pos_out = self.position_encoding(sam3_x_out).to(sam3_x_out.dtype)
            sam3_out.append(sam3_x_out)
            sam3_pos.append(sam3_pos_out)

            if self.sam2_convs is not None:
                sam2_x_out = self.sam2_convs[i](x)
                sam2_pos_out = self.position_encoding(sam2_x_out).to(sam2_x_out.dtype)
                sam2_out.append(sam2_x_out)
                sam2_pos.append(sam2_pos_out)
        return sam3_out, sam3_pos, sam2_out, sam2_pos
