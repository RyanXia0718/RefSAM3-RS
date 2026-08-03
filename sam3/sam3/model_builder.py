"""Image-model builder for the RefSAM3-RS full training pipeline."""

import gc
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from iopath.common.file_io import g_pathmgr

# ============================================================================
# Model submodule imports
# ============================================================================
from sam3.model.decoder import (
    TransformerDecoder,
    TransformerDecoderLayer_Ada,
)
from sam3.model.encoder import (
    TransformerEncoderFusion,
    TransformerEncoderLayer,
    TransformerEncoderLayer_Ada,
)
from sam3.model.geometry_encoders import SequenceGeometryEncoder
from sam3.model.maskformer_segmentation import PixelDecoder, UniversalSegmentationHead
from sam3.model.model_misc import (
    DotProductScoring,
    MLP,
    MultiheadAttentionWrapper as MultiheadAttention,
    TransformerWrapper,
)
from sam3.model.necks import Sam3DualViTDetNeck
from sam3.model.position_encoding import PositionEmbeddingSine
from sam3.model.sam3_image import Sam3Image
from sam3.model.text_encoder_ve_ada import create_adapter_text_encoder
from sam3.model.tokenizer_ve import SimpleTokenizer
from sam3.model.vitdet import ViT
from sam3.model.vl_combiner import SAM3VLBackbone


# ============================================================================
# Global configuration
# ============================================================================

# Base paths (auto-derived, no hardcoding needed)
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_SAM3_PKG_DIR = os.path.dirname(_CURRENT_DIR)  # sam3/ directory
_DEFAULT_ASSETS_DIR = os.path.join(_SAM3_PKG_DIR, "assets")

# Model dimension constants
D_MODEL = 256
D_FEEDFORWARD = 2048
N_HEADS = 8
DROPOUT = 0.1
RESOLUTION = 1008


@dataclass
class AdapterConfig:
    """Adapter configuration."""
    adapter_dim: int = 64
    adapter_heads: int = 4
    adapter_scale: float = 1.0

    def to_dict(self) -> Dict:
        return {"adapter_dim": self.adapter_dim, "adapter_heads": self.adapter_heads, "adapter_scale": self.adapter_scale}


@dataclass
class ModelConfig:
    """Configuration for the RefSAM3-RS image model."""
    # Basic settings
    device: str = "cuda"
    eval_mode: bool = True
    compile: bool = False

    enable_segmentation: bool = True

    adapter_config: AdapterConfig = field(default_factory=AdapterConfig)

    checkpoint_path: Optional[str] = None
    bpe_path: Optional[str] = None

    @property
    def compile_mode(self):
        return "default" if self.compile else None

    @property
    def adapter_dict(self) -> Dict:
        return self.adapter_config.to_dict()


# ============================================================================
# TF32 optimization
# ============================================================================

def _setup_tf32() -> None:
    """Enable TF32 acceleration for Ampere GPUs."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        if props.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

_setup_tf32()


# ============================================================================
# Model component construction functions (identical to original logic)
# ============================================================================

def _create_position_encoding(precompute_resolution=None):
    return PositionEmbeddingSine(num_pos_feats=D_MODEL, normalize=True, scale=None, temperature=10000, precompute_resolution=precompute_resolution)


def _create_vit_backbone(compile_mode=None, return_interm_layers=False):
    return ViT(
        img_size=RESOLUTION, pretrain_img_size=336, patch_size=14, embed_dim=1024, depth=32,
        num_heads=16, mlp_ratio=4.625, norm_layer="LayerNorm", drop_path_rate=DROPOUT,
        qkv_bias=True, use_abs_pos=True, tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31), rel_pos_blocks=(), use_rope=True,
        use_interp_rope=True, window_size=24, pretrain_use_cls_token=True,
        retain_cls_token=False, ln_pre=True, ln_post=False, return_interm_layers=return_interm_layers,
        bias_patch_embed=False, compile_mode=compile_mode,
    )


def _create_vit_neck(
    position_encoding,
    vit_backbone,
    textscale_config=None,
):
    return Sam3DualViTDetNeck(
        position_encoding=position_encoding, d_model=D_MODEL,
        scale_factors=[4.0, 2.0, 1.0, 0.5], trunk=vit_backbone,
        add_sam2_neck=False,
        textscale_config=textscale_config,
    )


def _create_vl_backbone(vit_neck, text_encoder):
    return SAM3VLBackbone(visual=vit_neck, text=text_encoder, scalp=1)


def _mha(batch_first=False):
    """Create standard multi-head attention."""
    return MultiheadAttention(num_heads=N_HEADS, dropout=DROPOUT, embed_dim=D_MODEL, batch_first=batch_first)


def _create_transformer_encoder(adapter_config) -> TransformerEncoderFusion:
    encoder_layer = TransformerEncoderLayer_Ada(
        activation="relu", d_model=D_MODEL, dim_feedforward=D_FEEDFORWARD, dropout=DROPOUT,
        pos_enc_at_attn=True, pos_enc_at_cross_attn_keys=False,
        pos_enc_at_cross_attn_queries=False, pre_norm=True,
        self_attention=_mha(batch_first=True), cross_attention=_mha(batch_first=True),
        adapter_config=adapter_config,
    )
    return TransformerEncoderFusion(
        layer=encoder_layer, num_layers=6, d_model=D_MODEL, num_feature_levels=1,
        frozen=False, use_act_checkpoint=True, add_pooled_text_to_img_feat=False,
        pool_text_with_mask=True,
    )


def _create_transformer_decoder(adapter_config) -> TransformerDecoder:
    decoder_layer = TransformerDecoderLayer_Ada(
        activation="relu", d_model=D_MODEL, dim_feedforward=D_FEEDFORWARD, dropout=DROPOUT,
        cross_attention=_mha(), n_heads=N_HEADS,
        use_text_cross_attention=True, adapter_config=adapter_config,
    )
    return TransformerDecoder(
        layer=decoder_layer, num_layers=6, num_queries=200, return_intermediate=True,
        box_refine=True, num_o2m_queries=0, dac=True, boxRPB="log", d_model=D_MODEL,
        frozen=False, interaction_layer=None, dac_use_selfatt_ln=True, resolution=RESOLUTION,
        stride=14, use_act_checkpoint=True, presence_token=True,
    )


def _create_dot_product_scoring():
    prompt_mlp = MLP(
        input_dim=D_MODEL, hidden_dim=D_FEEDFORWARD, output_dim=D_MODEL, num_layers=2,
        dropout=DROPOUT, residual=True, out_norm=nn.LayerNorm(D_MODEL),
    )
    return DotProductScoring(d_model=D_MODEL, d_proj=D_MODEL, prompt_mlp=prompt_mlp)


def _create_segmentation_head(compile_mode=None):
    pixel_decoder = PixelDecoder(
        num_upsampling_stages=3, interpolation_mode="nearest",
        hidden_dim=D_MODEL, compile_mode=compile_mode,
    )
    cross_attend_prompt = MultiheadAttention(num_heads=N_HEADS, dropout=0, embed_dim=D_MODEL)
    return UniversalSegmentationHead(
        hidden_dim=D_MODEL, upsampling_stages=3, aux_masks=False, presence_head=False,
        dot_product_scorer=None, act_ckpt=True, cross_attend_prompt=cross_attend_prompt,
        pixel_decoder=pixel_decoder,
    )


def _create_geometry_encoder():
    geo_pos_enc = _create_position_encoding()
    geo_layer = TransformerEncoderLayer(
        activation="relu", d_model=D_MODEL, dim_feedforward=D_FEEDFORWARD, dropout=DROPOUT,
        pos_enc_at_attn=False, pre_norm=True,
        self_attention=_mha(), pos_enc_at_cross_attn_queries=False,
        pos_enc_at_cross_attn_keys=True, cross_attention=_mha(),
    )
    return SequenceGeometryEncoder(
        pos_enc=geo_pos_enc, encode_boxes_as_points=False,
        points_direct_project=True, points_pool=True, points_pos_enc=True,
        boxes_direct_project=True, boxes_pool=True, boxes_pos_enc=True,
        d_model=D_MODEL, num_layers=3, layer=geo_layer, use_act_ckpt=True,
        add_cls=True, add_post_encode_proj=True,
    )


def _create_text_encoder(bpe_path: str, adapter_config):
    tokenizer = SimpleTokenizer(bpe_path=bpe_path)
    return create_adapter_text_encoder(
        d_model=D_MODEL,
        tokenizer=tokenizer,
        adapter_config=adapter_config,
        width=1024,
        heads=16,
        layers=24,
    )


def _create_vision_backbone(
    compile_mode=None,
    textscale_config=None,
) -> Sam3DualViTDetNeck:
    position_encoding = _create_position_encoding(precompute_resolution=RESOLUTION)
    use_textscale = bool(
        textscale_config and textscale_config.get("enabled", False)
    )
    vit_backbone = _create_vit_backbone(
        compile_mode=compile_mode,
        return_interm_layers=use_textscale,
    )
    return _create_vit_neck(
        position_encoding,
        vit_backbone,
        textscale_config=textscale_config,
    )


def _create_sam3_transformer(adapter_config) -> TransformerWrapper:
    encoder = _create_transformer_encoder(adapter_config=adapter_config)
    decoder = _create_transformer_decoder(adapter_config=adapter_config)
    return TransformerWrapper(encoder=encoder, decoder=decoder, d_model=D_MODEL)


# ============================================================================
# Unified weight loading logic
# ============================================================================

def _get_rank() -> int:
    """Get the rank of the current process."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _stagger_load():
    """Stagger multi-process loading to avoid IO spikes."""
    if dist.is_available() and dist.is_initialized():
        time.sleep((dist.get_rank() % 8) * 1.0)


def _load_raw_checkpoint(checkpoint_path: str) -> dict:
    """Load raw checkpoint from file and extract state_dict."""
    _stagger_load()
    print(f"[Rank {_get_rank()}] Loading checkpoint: {checkpoint_path}")
    with g_pathmgr.open(checkpoint_path, "rb") as f:
        ckpt = torch.load(f, map_location="cpu", weights_only=False)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        state_dict = ckpt["model"]
        del ckpt
    else:
        state_dict = ckpt
    gc.collect()
    return state_dict


def _filter_matching_weights(ckpt_state: dict, model_state: dict) -> dict:
    """Filter weights with matching shapes.

    When LoRA is injected before loading a pre-LoRA checkpoint, wrapped
    nn.Linear parameters move from foo.weight to
    foo.linear.weight. Preserve those base weights instead of leaving the
    wrapped linear layers at their fresh initialization.
    """
    filtered = {}
    lora_linear_remapped = 0
    for k, v in ckpt_state.items():
        model_key = k
        if model_key in model_state and v.shape == model_state[model_key].shape:
            filtered[model_key] = v
        elif model_key.endswith((".weight", ".bias")):
            base, suffix = model_key.rsplit(".", 1)
            lora_key = f"{base}.linear.{suffix}"
            if lora_key in model_state and v.shape == model_state[lora_key].shape:
                filtered[lora_key] = v
                lora_linear_remapped += 1
        elif model_key in model_state:
            print(
                f"  Skipping weight {k}: shape mismatch "
                f"{v.shape} vs {model_state[model_key].shape}"
            )
    if lora_linear_remapped:
        print(f"  Remapped {lora_linear_remapped} pre-LoRA Linear weights into LoRA wrappers")
    return filtered


def _remap_detector_keys(ckpt_state: dict) -> dict:
    """Remove the detector prefix used by the original SAM3 checkpoint."""
    remapped = {}
    for k, v in ckpt_state.items():
        if k.startswith("detector."):
            remapped[k.replace("detector.", "")] = v
    return remapped


def _sync_segmentation_head_adapters(model, from_adapter1=False, from_mask_predictor=True):
    """Synchronize segmentation_head adapter weights.
    
    Args:
        model: SAM3 model
        from_adapter1: If True, initialize adapter2 from adapter1
        from_mask_predictor: If True, initialize adapter1 and adapter2 from original mask_predictor
    """
    seg_head = getattr(model, 'segmentation_head', None)
    if seg_head is None:
        return
    if not (hasattr(seg_head, 'mask_predictor') and hasattr(seg_head, 'mask_predictor_adapter1')):
        return

    if from_mask_predictor:
        seg_head.mask_predictor_adapter1.load_state_dict(seg_head.mask_predictor.state_dict())
        seg_head.mask_predictor_adapter2.load_state_dict(seg_head.mask_predictor.state_dict())
        print("  ✓ Synced adapter1/adapter2 from mask_predictor")
    elif from_adapter1:
        seg_head.mask_predictor_adapter2.load_state_dict(seg_head.mask_predictor_adapter1.state_dict())
        print("  ✓ Synced adapter2 from adapter1")


def _zero_init_adapters(model):
    """Zero-initialize all Adapter module output projections (identity mapping)."""
    count = 0
    for name, module in model.named_modules():
        is_adapter = False
        if hasattr(module, 'up_proj') and isinstance(module.up_proj, nn.Linear):
            nn.init.zeros_(module.up_proj.weight)
            nn.init.zeros_(module.up_proj.bias)
            is_adapter = True
        if hasattr(module, 'up_conv') and isinstance(module.up_conv, nn.Conv2d):
            nn.init.zeros_(module.up_conv.weight)
            if module.up_conv.bias is not None:
                nn.init.zeros_(module.up_conv.bias)
            is_adapter = True
        if hasattr(module, 'mha_adapter') and isinstance(module.mha_adapter, nn.MultiheadAttention):
            if hasattr(module.mha_adapter, 'out_proj'):
                nn.init.zeros_(module.mha_adapter.out_proj.weight)
                nn.init.zeros_(module.mha_adapter.out_proj.bias)
            is_adapter = True
        if is_adapter:
            count += 1
    print(f"  ✓ Zero-initialized {count} Adapter outputs")


def load_checkpoint(model, cfg: ModelConfig):
    """Load either the pretrained SAM3 checkpoint or a Stage-1 checkpoint."""
    checkpoint_path = cfg.checkpoint_path
    if checkpoint_path is None:
        raise ValueError("checkpoint_path is required for RefSAM3-RS")

    ckpt_state = _load_raw_checkpoint(checkpoint_path)
    if any(k.startswith("detector.") for k in ckpt_state):
        ckpt_state = _remap_detector_keys(ckpt_state)

    # Preserve the released recipe's initialization sequence by recreating
    # the adapter text encoder immediately before checkpoint loading.
    _replace_text_encoder(model, cfg.bpe_path, cfg.adapter_dict)

    model_state = model.state_dict()
    filtered = _filter_matching_weights(ckpt_state, model_state)
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    has_adapter_in_ckpt = any("adapter" in key for key in filtered)
    print(
        f"  Loaded {len(filtered)} weights, "
        f"{len(missing)} missing, {len(unexpected)} unexpected"
    )

    should_init_adapter = not has_adapter_in_ckpt
    if should_init_adapter:
        print("  Performing Adapter zero-initialization...")
        _sync_segmentation_head_adapters(model, from_mask_predictor=True)
        _zero_init_adapters(model)
    else:
        print("  Skipping Adapter initialization (using checkpoint weights)")

    frozen, trainable = _freeze_non_adapter_params(model)
    print(f"  Freeze mode: {frozen} params frozen, {trainable} trainable")

    del ckpt_state, filtered
    gc.collect()


def _replace_text_encoder(model, bpe_path: str, adapter_config: dict):
    """Recreate the adapter text encoder before loading checkpoint weights."""
    old_state = None
    if hasattr(model.backbone, 'language_backbone') and model.backbone.language_backbone is not None:
        old_state = model.backbone.language_backbone.state_dict()

    new_encoder = _create_text_encoder(bpe_path, adapter_config=adapter_config)
    if old_state is not None:
        new_encoder.load_state_dict(old_state, strict=False)

    model.backbone.language_backbone = new_encoder

# ============================================================================
# Freeze logic
# ============================================================================

# Trainable parameter keyword whitelist
TRAINABLE_KEYWORDS = ('adapter', 'lora_')


def _freeze_non_adapter_params(model) -> tuple:
    """Freeze all non-adapter parameters, returns (frozen_count, trainable_count)."""
    frozen, trainable = 0, 0
    for name, param in model.named_parameters():
        if any(kw in name for kw in TRAINABLE_KEYWORDS):
            param.requires_grad = True
            trainable += 1
        else:
            param.requires_grad = False
            param.data.requires_grad = False
            frozen += 1
    return frozen, trainable


# ============================================================================
# Default BPE path
# ============================================================================

def _default_bpe_path() -> str:
    bpe = os.path.join(_DEFAULT_ASSETS_DIR, "bpe_simple_vocab_16e6.txt.gz")
    if os.path.exists(bpe):
        return bpe
    # Fallback to working directory
    return os.path.join(os.getcwd(), "assets", "bpe_simple_vocab_16e6.txt.gz")


# ============================================================================
# Main build entry: Image Model
# ============================================================================

def build_sam3_image_model(
    bpe_path=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    eval_mode=True,
    checkpoint_path=None,
    enable_segmentation=True,
    compile=False,
    adapter_config=None,
    visual_lora_config=None,
    textscale_config=None,
):
    """Build the RefSAM3-RS image model used by both training stages."""
    if bpe_path is None:
        bpe_path = _default_bpe_path()

    adapter_cfg = AdapterConfig()
    if adapter_config is not None:
        adapter_cfg = AdapterConfig(
            adapter_dim=adapter_config.get("adapter_dim", 64),
            adapter_heads=adapter_config.get("adapter_heads", 4),
            adapter_scale=adapter_config.get("adapter_scale", 1.0),
        )

    cfg = ModelConfig(
        device=device, eval_mode=eval_mode, compile=compile,
        enable_segmentation=enable_segmentation,
        adapter_config=adapter_cfg,
        checkpoint_path=checkpoint_path,
        bpe_path=bpe_path,
    )

    # ---- 1. Build model components ----
    vision_encoder = _create_vision_backbone(
        compile_mode=cfg.compile_mode,
        textscale_config=textscale_config,
    )
    text_encoder = _create_text_encoder(bpe_path, cfg.adapter_dict)
    backbone = _create_vl_backbone(vision_encoder, text_encoder)
    transformer = _create_sam3_transformer(adapter_config=cfg.adapter_dict)
    dot_prod_scoring = _create_dot_product_scoring()
    segmentation_head = (
        _create_segmentation_head(cfg.compile_mode)
        if cfg.enable_segmentation
        else None
    )
    input_geometry_encoder = _create_geometry_encoder()

    # ---- 2. Assemble model ----
    from sam3.train.matcher import BinaryHungarianMatcherV2
    matcher = BinaryHungarianMatcherV2(
        focal=True, cost_class=2.0, cost_bbox=5.0, cost_giou=2.0,
        alpha=0.25, gamma=2, stable=False,
    )
    model = Sam3Image(
        backbone=backbone, transformer=transformer,
        input_geometry_encoder=input_geometry_encoder,
        segmentation_head=segmentation_head, num_feature_levels=1,
        o2m_mask_predict=True, dot_prod_scoring=dot_prod_scoring,
        use_instance_query=False, multimask_output=True,
        matcher=matcher,
    )

    # ---- 3. Pre-load LoRA: apply before checkpoint so weights are found ----
    if visual_lora_config is not None:
        from sam3.model.lora import apply_lora
        r = visual_lora_config.get("lora_r", 8)
        alpha = visual_lora_config.get("lora_alpha", 16)
        dropout = visual_lora_config.get("lora_dropout", 0.0)
        vit_trunk = model.backbone.vision_backbone.trunk
        apply_lora(vit_trunk, r=r, alpha=alpha, dropout=dropout)
        lora_params = sum(p.numel() for n, p in vit_trunk.named_parameters() if "lora_" in n)
        print(f"Applied LoRA to ViT trunk (r={r}, alpha={alpha}): {lora_params:,} trainable params added")

    # ---- 4. Load weights ----
    load_checkpoint(model, cfg)

    # ---- 5. Device and mode ----
    if device == "cuda":
        model = model.cuda()
    if eval_mode:
        model.eval()

    return model
