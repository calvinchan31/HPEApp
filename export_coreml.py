#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Calvin Chan
#
# CoreML Export Script for HPE App
#
# Exports two CoreML models:
#   1. MobileNetProxy.mlpackage — distilled feature extractor (trained to match HMR2 token_out)
#   2. GVHMRStudent.mlpackage   — distilled GVHMR transformer (256-dim, 6 layers, 4 heads)
#
# Also exports gvhmr_stats.json with the normalization statistics needed on-device.
#
# Usage:
#     # First train the feature distillation model:
#     cd /home/calv0026/GVHMR8
#     python experiments/2026-02-22-iOSAPP/train_feature_distill.py
#
#     # Then export both models to CoreML:
#     python experiments/2026-02-22-iOSAPP/export_coreml.py
#
# Requirements:
#     pip install coremltools torch torchvision
#
# Model licensing: See ACKNOWLEDGMENTS.md for GVHMR, ViTPose, SMPL, and other third-party model licenses.

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(OUTPUT_DIR, "HPEApp", "Models")
RESOURCES_DIR = os.path.join(OUTPUT_DIR, "HPEApp")

STUDENT_CKPT = os.path.join(
    ROOT, "outputs/mocap_mixed_v1/mixed_student/checkpoints/e499-s377000.ckpt"
)
MEDIUM_CKPT = os.path.join(
    ROOT, "outputs/mocap_mixed_v1/mixed_student_medium/checkpoints/e499-s377000.ckpt"
)
ORIGINAL_CKPT = os.path.join(
    ROOT, "inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt"
)
DISTILLED_FEAT_CKPT = os.path.join(OUTPUT_DIR, "checkpoints", "mobilenet_distilled.pt")
WINDOW_SIZE = 16  # temporal window for inference


# =============================================================================
# 1. Export normalization statistics
# =============================================================================
def export_stats():
    """Export the MM_V1_AMASS_LOCAL_BEDLAM_CAM statistics to JSON."""
    from hmr4d.model.gvhmr.utils.stats_compose import MM_V1_AMASS_LOCAL_BEDLAM_CAM

    stats = {
        "mean": MM_V1_AMASS_LOCAL_BEDLAM_CAM["mean"],
        "std": MM_V1_AMASS_LOCAL_BEDLAM_CAM["std"],
        "pred_cam_mean": [1.0606, -0.0027, 0.2702],
        "pred_cam_std": [0.1784, 0.0956, 0.0764],
        "output_dim": 151,
        "window_size": WINDOW_SIZE,
        "description": {
            "body_pose_r6d": "indices 0-125 (21 joints * 6D rotation)",
            "betas": "indices 126-135 (10 shape params)",
            "global_orient_c_r6d": "indices 136-141 (camera-frame orientation, 6D)",
            "global_orient_gv_r6d": "indices 142-147 (gravity-aligned orientation, 6D)",
            "local_transl_vel": "indices 148-150 (local translation velocity)",
        },
    }

    out_path = os.path.join(RESOURCES_DIR, "gvhmr_stats.json")
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {out_path}")
    return stats


# =============================================================================
# 2. Export distilled MobileNet feature extractor
# =============================================================================
def export_mobilenet_proxy():
    """Export the distilled MobileNetV3-Small feature extractor to CoreML.

    If the distilled checkpoint exists, loads trained weights.
    Otherwise falls back to pretrained ImageNet weights (worse accuracy but still
    better than random, since appearance features have some body-pose signal).
    """
    import torchvision.models as models
    import coremltools as ct
    from train_feature_distill import MobileFeatureExtractor

    print("Exporting MobileNet Feature Extractor (1024-dim features)...")

    model = MobileFeatureExtractor()

    if os.path.exists(DISTILLED_FEAT_CKPT):
        print(f"  Loading distilled checkpoint: {DISTILLED_FEAT_CKPT}")
        ckpt = torch.load(DISTILLED_FEAT_CKPT, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded (epoch={ckpt['epoch']}, cos_sim={ckpt.get('cos_sim', 'N/A')})")
    else:
        print(f"  WARNING: Distilled checkpoint not found at {DISTILLED_FEAT_CKPT}")
        print(f"  Run train_feature_distill.py first for best results.")
        print(f"  Falling back to ImageNet-pretrained backbone with untrained adapter.")

    model.eval()

    dummy_img = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy_img)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="image", shape=dummy_img.shape)],
        minimum_deployment_target=ct.target.iOS16,
        convert_to="mlprogram",
    )

    out_path = os.path.join(MODELS_DIR, "MobileNetProxy.mlpackage")
    mlmodel.save(out_path)
    print(f"Saved MobileNet proxy to {out_path}")


# =============================================================================
# 3. Export GVHMR Student model
# =============================================================================
class GVHMRMobileWrapper(nn.Module):
    """
    Wrapper around NetworkEncoderRoPE for CoreML-friendly tracing.
    Fixed batch=1, fixed temporal window. All inputs are simple tensors.
    """

    def __init__(self, model, window_size=16):
        super().__init__()
        self.model = model
        self.window_size = window_size

    def forward(self, obs, f_cliffcam, f_cam_angvel, f_imgseq):
        """
        Args:
            obs:         (1, W, 17, 3) — normalized 2D keypoints with confidence
            f_cliffcam:  (1, W, 3)     — CLIFF camera params
            f_cam_angvel:(1, W, 6)     — ALREADY NORMALIZED camera angular velocity.
                         The iOS MotionManager normalizes using:
                         mean=[1,0,0,0,1,0], std=[0.001,0.1,0.1,0.1,0.001,0.1]
                         Do NOT double-normalize.
            f_imgseq:    (1, W, 1024)  — image features from MobileNet
        Returns:
            pred_x:   (1, W, 151) — encoded SMPL body params (normalized)
            pred_cam: (1, W, 3)   — camera params (s, tx, ty), already denormalized
        """
        length = torch.tensor([self.window_size], dtype=torch.long, device=obs.device)
        output = self.model(
            length,
            obs=obs,
            f_cliffcam=f_cliffcam,
            f_cam_angvel=f_cam_angvel,
            f_imgseq=f_imgseq,
        )
        return output["pred_x"], output["pred_cam"]


def export_gvhmr_student():
    """Export the distilled GVHMR student model to CoreML."""
    import coremltools as ct
    from hmr4d.network.gvhmr.relative_transformer import NetworkEncoderRoPE

    print("Exporting GVHMR Student (256-dim, 6 layers, 4 heads)...")

    # Build student with matching architecture
    student = NetworkEncoderRoPE(
        output_dim=151,
        max_len=120,
        cliffcam_dim=3,
        cam_angvel_dim=6,
        imgseq_dim=1024,
        latent_dim=256,
        num_layers=6,
        num_heads=4,
        mlp_ratio=4.0,
        pred_cam_dim=3,
        static_conf_dim=6,
        dropout=0.0,  # no dropout for inference
        avgbeta=True,
    )

    # Load fine-tuned weights
    if os.path.exists(STUDENT_CKPT):
        print(f"Loading checkpoint: {STUDENT_CKPT}")
        ckpt = torch.load(STUDENT_CKPT, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        new_sd = {}
        prefix = "pipeline.denoiser3d."
        for k, v in sd.items():
            if k.startswith(prefix):
                new_sd[k[len(prefix) :]] = v
        missing, unexpected = student.load_state_dict(new_sd, strict=False)
        print(f"  Loaded {len(new_sd)} weights (missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        print(f"WARNING: Checkpoint not found at {STUDENT_CKPT}, exporting with random weights")

    student.eval()

    # Wrap for mobile
    wrapper = GVHMRMobileWrapper(student, window_size=WINDOW_SIZE)
    wrapper.eval()

    # Trace
    W = WINDOW_SIZE
    dummy_obs = torch.rand(1, W, 17, 3)
    dummy_cliff = torch.rand(1, W, 3)
    dummy_angvel = torch.rand(1, W, 6)
    dummy_imgseq = torch.rand(1, W, 1024)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (dummy_obs, dummy_cliff, dummy_angvel, dummy_imgseq))

    # Convert to CoreML
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="obs", shape=(1, W, 17, 3)),
            ct.TensorType(name="f_cliffcam", shape=(1, W, 3)),
            ct.TensorType(name="f_cam_angvel", shape=(1, W, 6)),
            ct.TensorType(name="f_imgseq", shape=(1, W, 1024)),
        ],
        minimum_deployment_target=ct.target.iOS16,
        convert_to="mlprogram",
    )

    out_path = os.path.join(MODELS_DIR, "GVHMRStudent.mlpackage")
    mlmodel.save(out_path)
    print(f"Saved GVHMR Student to {out_path}")


# =============================================================================
# 4. Export GVHMR Medium model
# =============================================================================
def export_gvhmr_medium():
    """Export the medium GVHMR student model (384-dim, 8 layers, 6 heads) to CoreML."""
    import coremltools as ct
    from hmr4d.network.gvhmr.relative_transformer import NetworkEncoderRoPE

    print("Exporting GVHMR Medium (384-dim, 8 layers, 6 heads)...")

    medium = NetworkEncoderRoPE(
        output_dim=151,
        max_len=120,
        cliffcam_dim=3,
        cam_angvel_dim=6,
        imgseq_dim=1024,
        latent_dim=384,
        num_layers=8,
        num_heads=6,
        mlp_ratio=4.0,
        pred_cam_dim=3,
        static_conf_dim=6,
        dropout=0.0,
        avgbeta=True,
    )

    if os.path.exists(MEDIUM_CKPT):
        print(f"Loading checkpoint: {MEDIUM_CKPT}")
        ckpt = torch.load(MEDIUM_CKPT, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        new_sd = {}
        prefix = "pipeline.denoiser3d."
        for k, v in sd.items():
            if k.startswith(prefix):
                new_sd[k[len(prefix):]] = v
        missing, unexpected = medium.load_state_dict(new_sd, strict=False)
        print(f"  Loaded {len(new_sd)} weights (missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        print(f"WARNING: Checkpoint not found at {MEDIUM_CKPT}, exporting with random weights")

    medium.eval()

    wrapper = GVHMRMobileWrapper(medium, window_size=WINDOW_SIZE)
    wrapper.eval()

    W = WINDOW_SIZE
    dummy_obs = torch.rand(1, W, 17, 3)
    dummy_cliff = torch.rand(1, W, 3)
    dummy_angvel = torch.rand(1, W, 6)
    dummy_imgseq = torch.rand(1, W, 1024)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (dummy_obs, dummy_cliff, dummy_angvel, dummy_imgseq))

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="obs", shape=(1, W, 17, 3)),
            ct.TensorType(name="f_cliffcam", shape=(1, W, 3)),
            ct.TensorType(name="f_cam_angvel", shape=(1, W, 6)),
            ct.TensorType(name="f_imgseq", shape=(1, W, 1024)),
        ],
        minimum_deployment_target=ct.target.iOS16,
        convert_to="mlprogram",
    )

    out_path = os.path.join(MODELS_DIR, "GVHMRMedium.mlpackage")
    mlmodel.save(out_path)
    print(f"Saved GVHMR Medium to {out_path}")


# =============================================================================
# 5. Export original GVHMR model
# =============================================================================
def export_gvhmr_original():
    """Export the original GVHMR release model (512-dim, 12 layers, 8 heads) to CoreML."""
    import coremltools as ct
    from hmr4d.network.gvhmr.relative_transformer import NetworkEncoderRoPE

    print("Exporting GVHMR Original (512-dim, 12 layers, 8 heads)...")

    original = NetworkEncoderRoPE(
        output_dim=151,
        max_len=120,
        cliffcam_dim=3,
        cam_angvel_dim=6,
        imgseq_dim=1024,
        latent_dim=512,
        num_layers=12,
        num_heads=8,
        mlp_ratio=4.0,
        pred_cam_dim=3,
        static_conf_dim=6,
        dropout=0.0,
        avgbeta=True,
    )

    if os.path.exists(ORIGINAL_CKPT):
        print(f"Loading checkpoint: {ORIGINAL_CKPT}")
        ckpt = torch.load(ORIGINAL_CKPT, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        new_sd = {}
        prefix = "pipeline.denoiser3d."
        for k, v in sd.items():
            if k.startswith(prefix):
                new_sd[k[len(prefix):]] = v
        missing, unexpected = original.load_state_dict(new_sd, strict=False)
        print(f"  Loaded {len(new_sd)} weights (missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        print(f"WARNING: Checkpoint not found at {ORIGINAL_CKPT}, exporting with random weights")

    original.eval()

    wrapper = GVHMRMobileWrapper(original, window_size=WINDOW_SIZE)
    wrapper.eval()

    W = WINDOW_SIZE
    dummy_obs = torch.rand(1, W, 17, 3)
    dummy_cliff = torch.rand(1, W, 3)
    dummy_angvel = torch.rand(1, W, 6)
    dummy_imgseq = torch.rand(1, W, 1024)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (dummy_obs, dummy_cliff, dummy_angvel, dummy_imgseq))

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="obs", shape=(1, W, 17, 3)),
            ct.TensorType(name="f_cliffcam", shape=(1, W, 3)),
            ct.TensorType(name="f_cam_angvel", shape=(1, W, 6)),
            ct.TensorType(name="f_imgseq", shape=(1, W, 1024)),
        ],
        minimum_deployment_target=ct.target.iOS16,
        convert_to="mlprogram",
    )

    out_path = os.path.join(MODELS_DIR, "GVHMROriginal.mlpackage")
    mlmodel.save(out_path)
    print(f"Saved GVHMR Original to {out_path}")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESOURCES_DIR, exist_ok=True)

    print("=" * 60)
    print("HPE App — CoreML Export)")
    print("=" * 60)

    # 1. Stats
    export_stats()

    # 2. MobileNet proxy
    export_mobilenet_proxy()

    # 3. GVHMR student (small)
    export_gvhmr_student()

    # 4. GVHMR medium
    export_gvhmr_medium()

    # 5. GVHMR original
    export_gvhmr_original()

    print("\n" + "=" * 60)
    print("Export complete!")
    print(f"  Models:     {MODELS_DIR}/")
    print(f"  Statistics: {RESOURCES_DIR}/gvhmr_stats.json")
    print("=" * 60)
    print("\nExported GVHMR models:")
    print("  - GVHMRStudent.mlpackage  (small:    256-dim, 6 layers, 4 heads)")
    print("  - GVHMRMedium.mlpackage   (medium:   384-dim, 8 layers, 6 heads)")
    print("  - GVHMROriginal.mlpackage (original: 512-dim, 12 layers, 8 heads)")
    print("\nNext steps:")
    print("  1. Copy HPEApp/ to your Mac")
    print("  2. Open the Xcode project (or run xcodegen)")
    print("  3. Build and run on an iPhone")
