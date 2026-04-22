#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Calvin Chan
#
# Full end-to-end diagnostic: render SMPL meshes from both the full model
# and the student model on the tennis demo video, and compare visually.

Tests each pipeline stage:
  1. Student model pred_x → SMPL decode → mesh
  2. Full model pred_x → SMPL decode → mesh  
  3. Full model GT decode_dict → mesh
  4. Student with MobileNet features → mesh

Outputs: diagnose_output.mp4 with side-by-side rendering
"""

import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import cv2
import pickle

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle, axis_angle_to_matrix

DEMO_DIR = os.path.join(ROOT, "outputs/demo/tennis")
APP_DIR = os.path.dirname(os.path.abspath(__file__))
STUDENT_CKPT = os.path.join(ROOT, "outputs/mocap_mixed_v1/mixed_student/checkpoints/e499-s377000.ckpt")
STATS_PATH = os.path.join(APP_DIR, "GVHMRApp", "gvhmr_stats.json")
VIDEO_PATH = os.path.join(DEMO_DIR, "0_input_video.mp4")
SMPL_PATH = os.path.join(ROOT, "inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl")


def load_smpl():
    """Load SMPL model for mesh rendering."""
    with open(SMPL_PATH, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    return {
        "v_template": torch.tensor(np.array(data["v_template"]), dtype=torch.float32),
        "shapedirs": torch.tensor(np.array(data["shapedirs"][:, :, :10]), dtype=torch.float32),
        "J_regressor": torch.tensor(np.array(data["J_regressor"].todense()), dtype=torch.float32),
        "weights": torch.tensor(np.array(data["weights"]), dtype=torch.float32),
        "kintree": data["kintree_table"][0].astype(np.int64),
        "faces": np.array(data["f"], dtype=np.int32),
    }


def smpl_forward(smpl, body_pose_aa, global_orient_aa, betas):
    """
    Run SMPL forward pass.
    body_pose_aa: (21, 3) or (63,) axis-angle
    global_orient_aa: (3,) axis-angle
    betas: (10,)
    Returns: vertices (6890, 3)
    """
    body_pose_aa = body_pose_aa.reshape(21, 3)
    
    # Shape
    v_shaped = smpl["v_template"] + torch.einsum("vci,i->vc", smpl["shapedirs"], betas)
    
    # Joint locations
    J = torch.mm(smpl["J_regressor"], v_shaped)  # (24, 3)
    
    # Rotation matrices: root + 21 body + 2 identity hand = 24
    all_aa = torch.cat([global_orient_aa.unsqueeze(0), body_pose_aa, torch.zeros(2, 3)])  # (24, 3)
    rot_mats = axis_angle_to_matrix(all_aa)  # (24, 3, 3)
    
    # FK
    parents = smpl["kintree"].copy()
    parents[0] = -1
    
    world_R = [None] * 24
    world_t = [None] * 24
    rest_t = [None] * 24
    
    world_R[0] = rot_mats[0]
    world_t[0] = J[0]
    rest_t[0] = J[0]
    
    for i in range(1, 24):
        p = parents[i]
        local_t = J[i] - J[p]
        world_R[i] = world_R[p] @ rot_mats[i]
        world_t[i] = world_R[p] @ local_t + world_t[p]
        rest_t[i] = rest_t[p] + local_t
    
    # Deformation transforms
    T_list = []
    for i in range(24):
        t_trans = world_t[i] - world_R[i] @ rest_t[i]
        T_i = torch.cat([world_R[i], t_trans.unsqueeze(-1)], dim=-1)  # (3, 4)
        T_list.append(T_i)
    T = torch.stack(T_list)  # (24, 3, 4)
    
    # LBS
    W = smpl["weights"]  # (V, 24)
    T_flat = T.reshape(24, 12)
    T_blend = torch.mm(W, T_flat).reshape(-1, 3, 4)  # (V, 3, 4)
    
    v_homo = torch.cat([v_shaped, torch.ones(v_shaped.shape[0], 1)], dim=-1)  # (V, 4)
    verts = torch.bmm(T_blend, v_homo.unsqueeze(-1)).squeeze(-1)  # (V, 3)
    
    return verts


def project_verts_weak_perspective(verts, s, tx, ty, img_w, img_h):
    """Project vertices using weak-perspective camera."""
    x = s * verts[:, 0] + tx
    y = s * verts[:, 1] + ty
    px = ((x + 1) / 2 * img_w).numpy().astype(np.int32)
    py = ((y + 1) / 2 * img_h).numpy().astype(np.int32)
    return px, py, verts[:, 2].numpy()


def render_mesh_wireframe(frame, verts, faces, s, tx, ty, color=(200, 200, 200), alpha=0.5):
    """Render a SMPL mesh as wireframe overlay on frame."""
    h, w = frame.shape[:2]
    px, py, depth = project_verts_weak_perspective(verts, s, tx, ty, w, h)
    
    overlay = frame.copy()
    
    # Draw faces as filled triangles (sorted by depth for painter's algorithm)
    face_depths = []
    for f in faces:
        avg_depth = (depth[f[0]] + depth[f[1]] + depth[f[2]]) / 3
        face_depths.append(avg_depth)
    face_order = np.argsort(face_depths)[::-1]  # far to near
    
    for idx in face_order:
        f = faces[idx]
        pts = np.array([[px[f[0]], py[f[0]]], [px[f[1]], py[f[1]]], [px[f[2]], py[f[2]]]])
        if np.any(pts[:, 0] < 0) or np.any(pts[:, 0] >= w) or np.any(pts[:, 1] < 0) or np.any(pts[:, 1] >= h):
            continue
        cv2.fillPoly(overlay, [pts], color)
    
    result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    return result


def render_skeleton(frame, joints_3d, s, tx, ty, color=(0, 255, 0)):
    """Render 22-joint skeleton overlay."""
    h, w = frame.shape[:2]
    bones = [
        (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6),
        (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12),
        (9, 13), (9, 14), (12, 15), (13, 16), (14, 17),
        (16, 18), (17, 19), (18, 20), (19, 21),
    ]
    
    points = []
    for j in range(min(22, joints_3d.shape[0])):
        x_ndc = s * joints_3d[j, 0] + tx
        y_ndc = s * joints_3d[j, 1] + ty
        px = int((x_ndc + 1) / 2 * w)
        py = int((y_ndc + 1) / 2 * h)
        points.append((px, py))
    
    for a, b in bones:
        if a < len(points) and b < len(points):
            cv2.line(frame, points[a], points[b], color, 2)
    
    for pt in points:
        cv2.circle(frame, pt, 3, color, -1)
    
    return frame


def main():
    print("=" * 60)
    print("  Pipeline Diagnostic — Tennis Demo Video")
    print("=" * 60)
    
    # Load everything
    with open(STATS_PATH) as f:
        stats = json.load(f)
    mean = torch.tensor(stats["mean"]).float()
    std = torch.tensor(stats["std"]).float()
    pred_cam_mean = torch.tensor(stats["pred_cam_mean"]).float()
    pred_cam_std = torch.tensor(stats["pred_cam_std"]).float()
    
    gt = torch.load(os.path.join(DEMO_DIR, "hmr4d_results.pt"), map_location="cpu")
    smpl = load_smpl()
    
    # Full model outputs
    full_pred_x = gt["net_outputs"]["model_output"]["pred_x"][0]  # (F, 151)
    full_pred_cam = gt["net_outputs"]["model_output"]["pred_cam"][0]  # (F, 3)
    full_decode = gt["net_outputs"]["decode_dict"]
    gt_incam = gt["smpl_params_incam"]
    
    # Load student
    from hmr4d.network.gvhmr.relative_transformer import NetworkEncoderRoPE
    from hmr4d.utils.geo.hmr_cam import estimate_K, normalize_kp2d, compute_bbox_info_bedlam
    from hmr4d.utils.geo_transform import compute_cam_angvel
    
    student = NetworkEncoderRoPE(
        output_dim=151, max_len=120, cliffcam_dim=3, cam_angvel_dim=6,
        imgseq_dim=1024, latent_dim=256, num_layers=6, num_heads=4,
        mlp_ratio=4.0, pred_cam_dim=3, static_conf_dim=6, dropout=0.0, avgbeta=True)
    ckpt = torch.load(STUDENT_CKPT, map_location="cpu")
    sd = {k[len("pipeline.denoiser3d."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("pipeline.denoiser3d.")}
    student.load_state_dict(sd, strict=False)
    student.eval()
    
    # Preprocessed data
    bbx_data = torch.load(os.path.join(DEMO_DIR, "preprocess/bbx.pt"), map_location="cpu")
    vitpose = torch.load(os.path.join(DEMO_DIR, "preprocess/vitpose.pt"), map_location="cpu")
    vit_features = torch.load(os.path.join(DEMO_DIR, "preprocess/vit_features.pt"), map_location="cpu")
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    F = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    K_full = estimate_K(W, H).unsqueeze(0).repeat(F, 1, 1)
    R_w2c = torch.eye(3).unsqueeze(0).repeat(F, 1, 1)
    cam_angvel = compute_cam_angvel(R_w2c)
    obs = normalize_kp2d(vitpose, bbx_data["bbx_xys"])
    cliff_cam = compute_bbox_info_bedlam(bbx_data["bbx_xys"], K_full)
    
    # CRITICAL: Normalize cam_angvel (the pipeline does this but we were missing it!)
    cam_angvel_mean = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).float()
    cam_angvel_std = torch.tensor([0.001, 0.1, 0.1, 0.1, 0.001, 0.1]).float()
    cam_angvel = (cam_angvel - cam_angvel_mean) / cam_angvel_std
    print(f"cam_angvel after normalization: mean={cam_angvel.mean():.4f}, std={cam_angvel.std():.4f}")
    
    # Run student inference (all frames)
    print("Running student inference...")
    window = 16
    all_student_px = []
    all_student_cam = []
    with torch.no_grad():
        for start in range(0, F, window):
            end = min(start + window, F)
            actual = end - start
            o = obs[start:end]
            c = cliff_cam[start:end]
            a = cam_angvel[start:end]
            f_img = vit_features[start:end]
            if actual < window:
                pad = window - actual
                o = torch.cat([o, o[-1:].repeat(pad, 1, 1)])
                c = torch.cat([c, c[-1:].repeat(pad, 1)])
                a = torch.cat([a, a[-1:].repeat(pad, 1)])
                f_img = torch.cat([f_img, f_img[-1:].repeat(pad, 1)])
            out = student(torch.tensor([actual]),
                          obs=o.unsqueeze(0), f_cliffcam=c.unsqueeze(0),
                          f_cam_angvel=a.unsqueeze(0), f_imgseq=f_img.unsqueeze(0))
            all_student_px.append(out["pred_x"][0, :actual])
            all_student_cam.append(out["pred_cam"][0, :actual])
    
    student_px = torch.cat(all_student_px)  # (F, 151)
    student_cam = torch.cat(all_student_cam)  # (F, 3) — already denormalized
    
    # Decode student predictions
    student_x = student_px * std + mean
    student_bp_r6d = student_x[:, :126].reshape(-1, 21, 6)
    student_bp_R = rotation_6d_to_matrix(student_bp_r6d)
    student_bp_aa = matrix_to_axis_angle(student_bp_R).reshape(-1, 63)
    student_betas = student_x[:, 126:136]
    student_orient_r6d = student_x[:, 136:142]
    student_orient_R = rotation_6d_to_matrix(student_orient_r6d.reshape(-1, 6)).reshape(-1, 3, 3)
    student_orient_aa = matrix_to_axis_angle(student_orient_R)
    
    # Decode full model predictions  
    full_x = full_pred_x * std + mean
    full_bp_r6d = full_x[:, :126].reshape(-1, 21, 6)
    full_bp_R = rotation_6d_to_matrix(full_bp_r6d)
    full_bp_aa = matrix_to_axis_angle(full_bp_R).reshape(-1, 63)
    full_betas = full_x[:, 126:136]
    full_orient_r6d = full_x[:, 136:142]
    full_orient_R = rotation_6d_to_matrix(full_orient_r6d.reshape(-1, 6)).reshape(-1, 3, 3)
    full_orient_aa = matrix_to_axis_angle(full_orient_R)
    
    # GT decode (already in axis-angle from the stored decode_dict)
    gt_bp_aa = full_decode["body_pose"][0]  # (F, 63)
    gt_orient_aa = full_decode["global_orient"][0]  # (F, 3)
    gt_betas = gt_incam["betas"]  # (F, 10)
    gt_transl = gt_incam["transl"]  # (F, 3)
    
    # ====== KEY DIAGNOSTIC: Compare joint positions in 3D ======
    print("\nComputing 3D joint errors...")
    
    from hmr4d.utils.body_model.smpl_lite import SmplLite
    smpl_lite = SmplLite(SMPL_PATH)
    
    # Get J_regressor for computing joints from vertices
    J_reg = smpl["J_regressor"]  # (24, V)
    
    # Sample 5 frames for detailed comparison
    sample_frames = [0, F//4, F//2, 3*F//4, F-1]
    
    print(f"\n{'='*80}")
    print(f"  Per-frame 3D Analysis (sample frames)")
    print(f"{'='*80}")
    
    for fi in sample_frames:
        # Student SMPL
        s_beta = student_betas[fi] if student_betas[fi].dim() == 1 else student_betas.mean(0)
        sv = smpl_forward(smpl, student_bp_aa[fi], student_orient_aa[fi], s_beta)
        sj = torch.mm(J_reg, sv)[:22]  # (22, 3)
        
        # Full model (from pred_x decode)
        f_beta = full_betas[fi] if full_betas[fi].dim() == 1 else full_betas.mean(0)
        fv = smpl_forward(smpl, full_bp_aa[fi], full_orient_aa[fi], f_beta)
        fj = torch.mm(J_reg, fv)[:22]
        
        # GT (stored decode)
        gv = smpl_forward(smpl, gt_bp_aa[fi], gt_orient_aa[fi], gt_betas[fi])
        gj = torch.mm(J_reg, gv)[:22]
        
        # Errors
        err_sf = (sj - fj).norm(dim=-1).mean().item() * 1000  # mm
        err_sg = (sj - gj).norm(dim=-1).mean().item() * 1000
        err_fg = (fj - gj).norm(dim=-1).mean().item() * 1000
        
        print(f"\n  Frame {fi}:")
        print(f"    Student vs Full(pred_x):  {err_sf:.1f}mm")
        print(f"    Student vs GT(decode):    {err_sg:.1f}mm") 
        print(f"    Full(pred_x) vs GT(decode): {err_fg:.1f}mm")
        print(f"    Student cam: s={student_cam[fi,0]:.3f} tx={student_cam[fi,1]:.3f} ty={student_cam[fi,2]:.3f}")
        print(f"    Full cam:    s={full_pred_cam[fi,0]:.3f} tx={full_pred_cam[fi,1]:.3f} ty={full_pred_cam[fi,2]:.3f}")
    
    # ====== RENDER COMPARISON VIDEO ======
    print("\nRendering comparison video...")
    
    output_path = os.path.join(APP_DIR, "diagnose_output.mp4")
    frame_w = W
    frame_h = H
    out_w = frame_w * 3  # 3 panels side by side
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, 30, (out_w, frame_h))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    faces = smpl["faces"]
    
    for fi in range(min(F, 150)):  # first 150 frames for speed
        ret, frame = cap.read()
        if not ret:
            break
        
        # Panel 1: GT (full model decode_dict)
        panel_gt = frame.copy()
        try:
            gv = smpl_forward(smpl, gt_bp_aa[fi], gt_orient_aa[fi], gt_betas[fi])
            gs, gtx, gty = full_pred_cam[fi, 0].item(), full_pred_cam[fi, 1].item(), full_pred_cam[fi, 2].item()
            gj = torch.mm(J_reg, gv)[:22]
            panel_gt = render_skeleton(panel_gt, gj, gs, gtx, gty, color=(0, 255, 0))
        except Exception as e:
            cv2.putText(panel_gt, f"GT err: {e}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(panel_gt, "GT (Full Model)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Panel 2: Student + GT ViT features
        panel_student = frame.copy()
        try:
            sv = smpl_forward(smpl, student_bp_aa[fi], student_orient_aa[fi], student_betas[fi])
            ss, stx, sty = student_cam[fi, 0].item(), student_cam[fi, 1].item(), student_cam[fi, 2].item()
            sj = torch.mm(J_reg, sv)[:22]
            panel_student = render_skeleton(panel_student, sj, ss, stx, sty, color=(255, 100, 0))
        except Exception as e:
            cv2.putText(panel_student, f"Student err: {e}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(panel_student, "Student + GT ViT", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
        
        # Panel 3: Direct overlay comparison
        panel_both = frame.copy()
        try:
            panel_both = render_skeleton(panel_both, gj, gs, gtx, gty, color=(0, 255, 0))
            panel_both = render_skeleton(panel_both, sj, ss, stx, sty, color=(0, 0, 255))
        except:
            pass
        cv2.putText(panel_both, "Green=GT Red=Student", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add frame and cam info
        for panel in [panel_gt, panel_student, panel_both]:
            cv2.putText(panel, f"F#{fi}", (10, frame_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        combined = np.concatenate([panel_gt, panel_student, panel_both], axis=1)
        out.write(combined)
    
    cap.release()
    out.release()
    print(f"Saved: {output_path}")
    
    # ====== SUMMARY ======
    print(f"\n{'='*80}")
    print(f"  SUMMARY OF FINDINGS")
    print(f"{'='*80}")
    
    # Overall error
    n = min(student_bp_aa.shape[0], gt_bp_aa.shape[0])
    bp_diff = (student_bp_aa[:n] - gt_bp_aa[:n]).reshape(-1, 21, 3).norm(dim=-1)
    orient_diff = (student_orient_aa[:n] - gt_orient_aa[:n]).norm(dim=-1)
    
    print(f"\n  Student vs GT body_pose (axis-angle):")
    print(f"    Mean per-joint angle error: {bp_diff.mean():.4f} rad ({bp_diff.mean() * 180/3.14159:.1f} deg)")
    print(f"    Max per-joint angle error:  {bp_diff.max():.4f} rad ({bp_diff.max() * 180/3.14159:.1f} deg)")
    
    JOINTS = ["L_Hip","R_Hip","Spine1","L_Knee","R_Knee","Spine2","L_Ankle","R_Ankle","Spine3",
              "L_Foot","R_Foot","Neck","L_Collar","R_Collar","Head","L_Shoulder","R_Shoulder",
              "L_Elbow","R_Elbow","L_Wrist","R_Wrist"]
    
    print(f"\n  Worst 5 joints (by mean error):")
    joint_errors = [(JOINTS[j], bp_diff[:, j].mean().item()) for j in range(21)]
    joint_errors.sort(key=lambda x: -x[1])
    for name, err in joint_errors[:5]:
        print(f"    {name:<15} {err:.4f} rad ({err*180/3.14159:.1f} deg)")
    
    print(f"\n  Global orient error: mean={orient_diff.mean():.4f} rad ({orient_diff.mean()*180/3.14159:.1f} deg)")
    
    # Student vs Full model pred_x
    full_bp_diff = (student_bp_aa[:n] - full_bp_aa[:n]).reshape(-1, 21, 3).norm(dim=-1)
    print(f"\n  Student vs Full model body_pose:")
    print(f"    Mean per-joint angle error: {full_bp_diff.mean():.4f} rad ({full_bp_diff.mean()*180/3.14159:.1f} deg)")
    
    # Full model pred_x vs GT decode
    full_gt_diff = (full_bp_aa[:n] - gt_bp_aa[:n]).reshape(-1, 21, 3).norm(dim=-1)
    print(f"\n  Full model pred_x decode vs GT decode_dict:")
    print(f"    Mean per-joint angle error: {full_gt_diff.mean():.4f} rad ({full_gt_diff.mean()*180/3.14159:.1f} deg)")
    
    # Camera comparison
    cam_diff = (student_cam[:n] - full_pred_cam[:n]).abs()
    print(f"\n  Camera param differences (Student vs Full):")
    print(f"    Scale s:  mean_diff={cam_diff[:, 0].mean():.4f}")
    print(f"    Trans tx: mean_diff={cam_diff[:, 1].mean():.4f}")
    print(f"    Trans ty: mean_diff={cam_diff[:, 2].mean():.4f}")
    
    # Check: is the student using a different betas?
    beta_diff = (student_betas[:n] - gt_betas[:n]).abs()
    print(f"\n  Betas difference: MAE={beta_diff.mean():.4f}")
    print(f"    Student betas (mean): {student_betas.mean(0)[:5].tolist()}")
    print(f"    GT betas (mean):      {gt_betas.mean(0)[:5].tolist()}")


if __name__ == "__main__":
    main()
