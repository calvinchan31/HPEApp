#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 GVHMR iOS App Contributors
""""
Test the GVHMR Student model on the tennis demo video.

 Compares three scenarios:
  A) Student model with ground-truth HMR2 ViT features (from original preprocessing)
  B) Student model with distilled MobileNet features
  C) Ground-truth results from the full GVHMR model (hmr4d_results.pt)

Visualizes which body joints are "stuck" and diagnoses the stuck-hands issue.

Usage:
    cd /home/calv0026/GVHMR8
    conda activate gvhmr
    python experiments/2026-02-22-iOSAPP/test_demo_video.py
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle, matrix_to_rotation_6d
from hmr4d.utils.geo.hmr_cam import estimate_K, normalize_kp2d, compute_bbox_info_bedlam
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.network.gvhmr.relative_transformer import NetworkEncoderRoPE

# Paths
DEMO_DIR = os.path.join(ROOT, "outputs/demo/tennis")
PREPROC_DIR = os.path.join(DEMO_DIR, "preprocess")
APP_DIR = os.path.dirname(os.path.abspath(__file__))
STUDENT_CKPT = os.path.join(ROOT, "outputs/mocap_mixed_v1/mixed_student/checkpoints/e499-s377000.ckpt")
DISTILLED_FEAT_CKPT = os.path.join(APP_DIR, "checkpoints", "mobilenet_distilled.pt")
STATS_PATH = os.path.join(APP_DIR, "GVHMRApp", "gvhmr_stats.json")
VIDEO_PATH = os.path.join(DEMO_DIR, "0_input_video.mp4")

# 21 SMPL body joints (excluding root)
JOINT_NAMES = [
    "L_Hip", "R_Hip", "Spine1",
    "L_Knee", "R_Knee", "Spine2",
    "L_Ankle", "R_Ankle", "Spine3",
    "L_Foot", "R_Foot", "Neck",
    "L_Collar", "R_Collar", "Head",
    "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist",
]


def load_student_model():
    """Load the student model with trained weights."""
    student = NetworkEncoderRoPE(
        output_dim=151, max_len=120, cliffcam_dim=3, cam_angvel_dim=6,
        imgseq_dim=1024, latent_dim=256, num_layers=6, num_heads=4,
        mlp_ratio=4.0, pred_cam_dim=3, static_conf_dim=6, dropout=0.0,
        avgbeta=True,
    )
    ckpt = torch.load(STUDENT_CKPT, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)
    new_sd = {}
    prefix = "pipeline.denoiser3d."
    for k, v in sd.items():
        if k.startswith(prefix):
            new_sd[k[len(prefix):]] = v
    missing, unexpected = student.load_state_dict(new_sd, strict=False)
    print(f"Student model loaded: {len(new_sd)} weights (missing={len(missing)}, unexpected={len(unexpected)})")
    if missing:
        print(f"  Missing: {missing}")
    student.eval()
    return student


def load_mobilenet():
    """Load the distilled MobileNet feature extractor."""
    sys.path.insert(0, APP_DIR)
    from train_feature_distill import MobileFeatureExtractor
    model = MobileFeatureExtractor()
    ckpt = torch.load(DISTILLED_FEAT_CKPT, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"MobileNet loaded: epoch={ckpt['epoch']}, cos_sim={ckpt.get('cos_sim', 'N/A')}")
    return model


def load_stats():
    """Load normalization statistics."""
    with open(STATS_PATH) as f:
        stats = json.load(f)
    mean = torch.tensor(stats["mean"]).float()
    std = torch.tensor(stats["std"]).float()
    pred_cam_mean = torch.tensor(stats["pred_cam_mean"]).float()
    pred_cam_std = torch.tensor(stats["pred_cam_std"]).float()
    return mean, std, pred_cam_mean, pred_cam_std


def load_preprocessed_data():
    """Load the ground-truth preprocessed data from the tennis demo."""
    bbx_data = torch.load(os.path.join(PREPROC_DIR, "bbx.pt"), map_location="cpu")
    bbx_xys = bbx_data["bbx_xys"]       # (F, 3)
    vitpose = torch.load(os.path.join(PREPROC_DIR, "vitpose.pt"), map_location="cpu")  # (F, 17, 3)
    vit_features = torch.load(os.path.join(PREPROC_DIR, "vit_features.pt"), map_location="cpu")  # (F, 1024)
    gt_results = torch.load(os.path.join(DEMO_DIR, "hmr4d_results.pt"), map_location="cpu")

    # Video dimensions
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"Video: {length}x{width}x{height}")

    # Camera intrinsics (same as demo.py)
    K_fullimg = estimate_K(width, height).unsqueeze(0).repeat(length, 1, 1)  # (F, 3, 3)

    # Static camera → identity rotation → cam_angvel
    R_w2c = torch.eye(3).unsqueeze(0).repeat(length, 1, 1)  # (F, 3, 3)
    cam_angvel = compute_cam_angvel(R_w2c)  # (F, 6)

    # CRITICAL: Normalize cam_angvel (the training pipeline does this)
    cam_angvel_mean = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).float()
    cam_angvel_std = torch.tensor([0.001, 0.1, 0.1, 0.1, 0.001, 0.1]).float()
    cam_angvel = (cam_angvel - cam_angvel_mean) / cam_angvel_std

    # Normalize keypoints (matching the pipeline)
    obs = normalize_kp2d(vitpose, bbx_xys)  # (F, 17, 3)

    # CLIFF camera params
    cliff_cam = compute_bbox_info_bedlam(bbx_xys, K_fullimg)  # (F, 3)

    print(f"  bbx_xys: {bbx_xys.shape}")
    print(f"  vitpose (kp2d): {vitpose.shape}")
    print(f"  vit_features: {vit_features.shape}")
    print(f"  obs (normalized): {obs.shape}")
    print(f"  cliff_cam: {cliff_cam.shape}")
    print(f"  cam_angvel: {cam_angvel.shape}")

    return {
        "obs": obs,
        "cliff_cam": cliff_cam,
        "cam_angvel": cam_angvel,
        "vit_features": vit_features,
        "K_fullimg": K_fullimg,
        "bbx_xys": bbx_xys,
        "vitpose": vitpose,
        "gt_results": gt_results,
        "length": length,
        "width": width,
        "height": height,
    }


def extract_mobilenet_features(model, video_path, bbx_xys, max_frames=None):
    """Extract MobileNet features from video frames, mimicking iOS app preprocessing."""
    IMAGE_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGE_STD = np.array([0.229, 0.224, 0.225])

    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        n_frames = min(n_frames, max_frames)

    features = []
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Crop using bbx_xys (matching HMR2 preprocessing)
        cx, cy, s = bbx_xys[i].numpy()
        hs = s * 1.2 / 2
        src = np.array([
            [cx - hs, cy - hs],
            [cx + hs, cy - hs],
            [cx, cy],
        ], dtype=np.float32)
        dst = np.array([
            [0, 0],
            [223, 0],
            [111.5, 111.5],
        ], dtype=np.float32)
        A = cv2.getAffineTransform(src, dst)
        crop = cv2.warpAffine(frame_rgb, A, (224, 224), flags=cv2.INTER_LINEAR)

        # Normalize
        crop = crop.astype(np.float32) / 255.0
        crop = (crop - IMAGE_MEAN) / IMAGE_STD
        crop_tensor = torch.from_numpy(crop).permute(2, 0, 1).float().unsqueeze(0)

        with torch.no_grad():
            feat = model(crop_tensor)  # (1, 1024)
        features.append(feat.squeeze(0))

    cap.release()
    features = torch.stack(features)  # (F, 1024)
    return features


@torch.no_grad()
def run_student_inference(model, obs, cliff_cam, cam_angvel, f_imgseq, window_size=16):
    """Run sliding-window inference matching the student model."""
    F_total = obs.shape[0]
    all_pred_x = []
    all_pred_cam = []

    # Process in overlapping windows
    for start in range(0, F_total, window_size):
        end = min(start + window_size, F_total)
        actual_len = end - start

        # Pad to window_size if needed
        obs_w = obs[start:end]
        cliff_w = cliff_cam[start:end]
        angvel_w = cam_angvel[start:end]
        imgseq_w = f_imgseq[start:end]

        if actual_len < window_size:
            pad = window_size - actual_len
            obs_w = torch.cat([obs_w, obs_w[-1:].repeat(pad, 1, 1)])
            cliff_w = torch.cat([cliff_w, cliff_w[-1:].repeat(pad, 1)])
            angvel_w = torch.cat([angvel_w, angvel_w[-1:].repeat(pad, 1)])
            imgseq_w = torch.cat([imgseq_w, imgseq_w[-1:].repeat(pad, 1)])

        # Add batch dim
        length = torch.tensor([actual_len])
        output = model(
            length,
            obs=obs_w.unsqueeze(0),
            f_cliffcam=cliff_w.unsqueeze(0),
            f_cam_angvel=angvel_w.unsqueeze(0),
            f_imgseq=imgseq_w.unsqueeze(0),
        )

        all_pred_x.append(output["pred_x"][0, :actual_len])
        all_pred_cam.append(output["pred_cam"][0, :actual_len])

    pred_x = torch.cat(all_pred_x, dim=0)  # (F, 151)
    pred_cam = torch.cat(all_pred_cam, dim=0)  # (F, 3)
    return pred_x, pred_cam


def decode_body_pose(pred_x_norm, mean, std):
    """Decode normalized output to body pose parameters."""
    pred_x = pred_x_norm * std + mean

    body_pose_r6d = pred_x[:, :126].reshape(-1, 21, 6)  # (F, 21, 6)
    betas = pred_x[:, 126:136]  # (F, 10)
    global_orient_r6d = pred_x[:, 136:142]  # (F, 6)
    global_orient_gv_r6d = pred_x[:, 142:148]  # (F, 6)
    local_transl_vel = pred_x[:, 148:151]  # (F, 3)

    # Convert to rotation matrices
    body_pose_R = rotation_6d_to_matrix(body_pose_r6d)  # (F, 21, 3, 3)
    global_orient_R = rotation_6d_to_matrix(global_orient_r6d.reshape(-1, 6)).reshape(-1, 3, 3)

    # Convert to axis-angle for SMPL
    body_pose_aa = matrix_to_axis_angle(body_pose_R).reshape(-1, 63)
    global_orient_aa = matrix_to_axis_angle(global_orient_R)

    return {
        "body_pose_r6d": body_pose_r6d,
        "body_pose_R": body_pose_R,
        "body_pose_aa": body_pose_aa,
        "betas": betas,
        "global_orient_r6d": global_orient_r6d,
        "global_orient_R": global_orient_R,
        "global_orient_aa": global_orient_aa,
        "local_transl_vel": local_transl_vel,
    }


def analyze_joint_variation(decoded, label=""):
    """Analyze per-joint rotation variation across frames."""
    body_pose_aa = decoded["body_pose_aa"].reshape(-1, 21, 3)  # (F, 21, 3)
    F_total = body_pose_aa.shape[0]

    print(f"\n{'='*60}")
    print(f"  Joint Rotation Analysis: {label}")
    print(f"  Frames: {F_total}")
    print(f"{'='*60}")
    print(f"  {'Joint':<15} {'Mean Angle':<12} {'Std':<10} {'Range':<12} {'Movement'}")
    print(f"  {'-'*65}")

    for j in range(21):
        angles = body_pose_aa[:, j].norm(dim=-1)  # rotation magnitude per frame
        mean_angle = angles.mean().item()
        std_angle = angles.std().item()
        range_angle = (angles.max() - angles.min()).item()

        # Also compute frame-to-frame change
        if F_total > 1:
            diffs = (body_pose_aa[1:, j] - body_pose_aa[:-1, j]).norm(dim=-1)
            movement = diffs.mean().item()
        else:
            movement = 0

        stuck = "STUCK" if std_angle < 0.01 and movement < 0.005 else ""
        print(f"  {JOINT_NAMES[j]:<15} {mean_angle:>8.4f}    {std_angle:>8.4f}  {range_angle:>8.4f}    {movement:>8.5f}  {stuck}")

    # Global orient
    go_aa = decoded["global_orient_aa"]
    angles = go_aa.norm(dim=-1)
    print(f"\n  {'Global Orient':<15} mean={angles.mean():.4f}  std={angles.std():.4f}  range={angles.max()-angles.min():.4f}")

    # Translation velocity
    tv = decoded["local_transl_vel"]
    print(f"  {'Transl Vel':<15} mean_mag={tv.norm(dim=-1).mean():.4f}  std={tv.norm(dim=-1).std():.4f}")


def compare_features(gt_features, mobile_features):
    """Compare ground-truth HMR2 features with distilled MobileNet features."""
    n = min(gt_features.shape[0], mobile_features.shape[0])
    gt = gt_features[:n]
    mob = mobile_features[:n]

    cos_sim = torch.nn.functional.cosine_similarity(gt, mob, dim=1)
    mse = ((gt - mob) ** 2).mean(dim=1)

    print(f"\n{'='*60}")
    print(f"  Feature Comparison: GT ViT vs Distilled MobileNet")
    print(f"{'='*60}")
    print(f"  Cosine Similarity:  mean={cos_sim.mean():.4f}  std={cos_sim.std():.4f}  min={cos_sim.min():.4f}  max={cos_sim.max():.4f}")
    print(f"  MSE:                mean={mse.mean():.4f}  std={mse.std():.4f}")
    print(f"  Feature magnitude (GT):    {gt.norm(dim=1).mean():.4f}")
    print(f"  Feature magnitude (Mob):   {mob.norm(dim=1).mean():.4f}")


def compare_predictions(pred_x_A, pred_x_B, mean, std, label_A="A", label_B="B"):
    """Compare predictions between two runs."""
    x_A = pred_x_A * std + mean
    x_B = pred_x_B * std + mean

    n = min(x_A.shape[0], x_B.shape[0])
    x_A = x_A[:n]
    x_B = x_B[:n]

    # Per-component comparison
    components = [
        ("body_pose_r6d[0:126]", 0, 126),
        ("betas[126:136]", 126, 136),
        ("orient_c_r6d[136:142]", 136, 142),
        ("orient_gv_r6d[142:148]", 142, 148),
        ("transl_vel[148:151]", 148, 151),
    ]

    print(f"\n{'='*60}")
    print(f"  Prediction Comparison: {label_A} vs {label_B}")
    print(f"{'='*60}")
    for name, s, e in components:
        diff = (x_A[:, s:e] - x_B[:, s:e]).abs()
        cos = torch.nn.functional.cosine_similarity(
            x_A[:, s:e].reshape(n, -1), x_B[:, s:e].reshape(n, -1), dim=1
        )
        print(f"  {name:<30} MAE={diff.mean():.4f}  Max={diff.max():.4f}  CosSim={cos.mean():.4f}")


def render_comparison_video(data, decoded_A, decoded_B, decoded_GT, pred_cam_A, pred_cam_B,
                             mean, std, pred_cam_mean, pred_cam_std,
                             output_path="test_comparison.mp4"):
    """Render a side-by-side comparison video showing 2D projections from all 3 methods."""
    from hmr4d.utils.smplx_utils import make_smplx

    cap = cv2.VideoCapture(VIDEO_PATH)
    width = data["width"]
    height = data["height"]
    n_frames = min(data["length"], decoded_A["body_pose_aa"].shape[0])

    # Simple: show joint angles as bar chart overlay
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (width * 2, height))

    print(f"\nRendering comparison to {output_path}...")
    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Left: original video with info
        left = frame.copy()

        # Draw 2D keypoints from vitpose
        kp2d = data["vitpose"][i].numpy()  # (17, 3)
        for j in range(17):
            if kp2d[j, 2] > 0.3:
                x, y = int(kp2d[j, 0]), int(kp2d[j, 1])
                cv2.circle(left, (x, y), 4, (0, 255, 0), -1)

        # Right: joint angle comparison
        right = np.zeros_like(frame)

        # Show per-joint angle magnitude for student-A vs student-B
        bp_A = decoded_A["body_pose_aa"][i].reshape(21, 3).norm(dim=-1).numpy()
        bp_B = decoded_B["body_pose_aa"][min(i, decoded_B["body_pose_aa"].shape[0]-1)].reshape(21, 3).norm(dim=-1).numpy()

        bar_h = height // 22
        for j in range(21):
            y_top = j * bar_h + 2
            # Bar for A (blue)
            bar_w_A = int(min(bp_A[j] / 3.0 * width * 0.4, width * 0.45))
            cv2.rectangle(right, (5, y_top), (5 + bar_w_A, y_top + bar_h // 2 - 1), (255, 150, 50), -1)
            # Bar for B (red)
            bar_w_B = int(min(bp_B[j] / 3.0 * width * 0.4, width * 0.45))
            cv2.rectangle(right, (5, y_top + bar_h // 2), (5 + bar_w_B, y_top + bar_h - 1), (50, 50, 255), -1)
            # Label
            cv2.putText(right, JOINT_NAMES[j], (width // 2 + 10, y_top + bar_h - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        # Legend
        cv2.putText(right, "Blue=GT_feat  Red=Mobile_feat", (10, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(left, f"Frame {i}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        combined = np.concatenate([left, right], axis=1)
        out.write(combined)

    cap.release()
    out.release()
    print(f"Saved comparison video: {output_path}")


def main():
    print("=" * 60)
    print("  GVHMR Student Model — Tennis Demo Test")
    print("=" * 60)

    # Load everything
    mean, std, pred_cam_mean, pred_cam_std = load_stats()
    data = load_preprocessed_data()
    student = load_student_model()

    print("\n" + "=" * 60)
    print("  [A] Running student with GT HMR2 ViT features")
    print("=" * 60)
    pred_x_A, pred_cam_A = run_student_inference(
        student, data["obs"], data["cliff_cam"], data["cam_angvel"], data["vit_features"]
    )
    decoded_A = decode_body_pose(pred_x_A, mean, std)
    analyze_joint_variation(decoded_A, "Student + GT ViT Features")

    print("\n" + "=" * 60)
    print("  [B] Running student with Distilled MobileNet features")
    print("=" * 60)
    mobilenet = load_mobilenet()
    print("Extracting MobileNet features from video...")
    mobile_features = extract_mobilenet_features(mobilenet, VIDEO_PATH, data["bbx_xys"])
    print(f"  MobileNet features: {mobile_features.shape}")

    compare_features(data["vit_features"], mobile_features)

    pred_x_B, pred_cam_B = run_student_inference(
        student, data["obs"], data["cliff_cam"], data["cam_angvel"], mobile_features
    )
    decoded_B = decode_body_pose(pred_x_B, mean, std)
    analyze_joint_variation(decoded_B, "Student + Distilled MobileNet")

    # Compare predictions
    compare_predictions(pred_x_A, pred_x_B, mean, std, "GT ViT", "Distilled Mobile")

    # Also check: what if we pass zeros for f_imgseq?
    print("\n" + "=" * 60)
    print("  [C] Running student with ZERO features (no image features)")
    print("=" * 60)
    zero_features = torch.zeros_like(data["vit_features"])
    pred_x_C, pred_cam_C = run_student_inference(
        student, data["obs"], data["cliff_cam"], data["cam_angvel"], zero_features
    )
    decoded_C = decode_body_pose(pred_x_C, mean, std)
    analyze_joint_variation(decoded_C, "Student + ZERO Features")

    compare_predictions(pred_x_A, pred_x_C, mean, std, "GT ViT", "Zero Features")

    # Check GT results
    print("\n" + "=" * 60)
    print("  [GT] Analyzing ground-truth results (full GVHMR)")
    print("=" * 60)
    gt = data["gt_results"]
    print("  GT keys:", list(gt.keys()))
    for k, v in gt.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k}: {v.shape}")
        elif isinstance(v, dict):
            print(f"    {k}:")
            for kk, vv in v.items():
                if isinstance(vv, torch.Tensor):
                    print(f"      {kk}: {vv.shape}")

    # Check GT body pose
    if "smpl_params_incam" in gt:
        gt_bp = gt["smpl_params_incam"]["body_pose"]  # (F, 63)
        gt_bp_reshaped = gt_bp.reshape(-1, 21, 3)
        print(f"\n  GT body_pose shape: {gt_bp.shape}")
        print(f"\n  {'Joint':<15} {'Mean Angle':<12} {'Std':<10} {'Range':<12} {'Movement'}")
        print(f"  {'-'*65}")
        for j in range(21):
            angles = gt_bp_reshaped[:, j].norm(dim=-1)
            mean_a = angles.mean().item()
            std_a = angles.std().item()
            range_a = (angles.max() - angles.min()).item()
            if gt_bp_reshaped.shape[0] > 1:
                diffs = (gt_bp_reshaped[1:, j] - gt_bp_reshaped[:-1, j]).norm(dim=-1)
                movement = diffs.mean().item()
            else:
                movement = 0
            print(f"  {JOINT_NAMES[j]:<15} {mean_a:>8.4f}    {std_a:>8.4f}  {range_a:>8.4f}    {movement:>8.5f}")

    # Also compare decoded pred_cam
    print("\n" + "=" * 60)
    print("  pred_cam Analysis")
    print("=" * 60)
    # pred_cam is already denormalized inside the model
    cam_A = pred_cam_A
    cam_B = pred_cam_B
    print(f"  [A] GT feat:    s={cam_A[:, 0].mean():.4f}  tx={cam_A[:, 1].mean():.4f}  ty={cam_A[:, 2].mean():.4f}")
    print(f"  [B] Mobile:     s={cam_B[:, 0].mean():.4f}  tx={cam_B[:, 1].mean():.4f}  ty={cam_B[:, 2].mean():.4f}")
    print(f"  [A] std:        s={cam_A[:, 0].std():.4f}  tx={cam_A[:, 1].std():.4f}  ty={cam_A[:, 2].std():.4f}")
    print(f"  [B] std:        s={cam_B[:, 0].std():.4f}  tx={cam_B[:, 1].std():.4f}  ty={cam_B[:, 2].std():.4f}")

    # Also check the iOS-like decoding
    print("\n" + "=" * 60)
    print("  iOS Decoding Check (checking r6d validity)")
    print("=" * 60)
    # Check if r6d values produce valid rotation matrices
    for label, decoded in [("A-GT", decoded_A), ("B-Mobile", decoded_B)]:
        bpr6d = decoded["body_pose_r6d"]  # (F, 21, 6)
        R = decoded["body_pose_R"]  # (F, 21, 3, 3)
        # Check if R is valid rotation (det=1, R^T R = I)
        det = torch.det(R.reshape(-1, 3, 3))
        RTR = torch.bmm(R.reshape(-1, 3, 3).transpose(-1, -2), R.reshape(-1, 3, 3))
        I_err = (RTR - torch.eye(3)).norm(dim=(-1, -2))
        print(f"  [{label}] det: mean={det.mean():.6f} std={det.std():.6f}")
        print(f"  [{label}] R^T R - I error: mean={I_err.mean():.6f} max={I_err.max():.6f}")

    # Render comparison video
    output_video = os.path.join(APP_DIR, "test_comparison.mp4")
    render_comparison_video(data, decoded_A, decoded_B, None, pred_cam_A, pred_cam_B,
                           mean, std, pred_cam_mean, pred_cam_std, output_video)

    print("\n" + "=" * 60)
    print("  DONE. Check results above and comparison video.")
    print("=" * 60)


if __name__ == "__main__":
    main()
