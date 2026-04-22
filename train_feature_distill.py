#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Calvin Chan
#
# Feature Distillation: Train MobileNetV3-Small to mimic HMR2's token_out (1024-dim).
#
# Uses pre-computed HMR2 features from 3DPW dataset as ground truth,
# and source images cropped with bbox info to train the mobile model.
#
# The trained model replaces the untrained MobileNet proxy in the iOS app.
#
# Usage:
#     cd /home/calv0026/GVHMR8
#     conda activate gvhmr
#     python experiments/2026-02-22-iOSAPP/train_feature_distill.py
#
# Model licensing: See ACKNOWLEDGMENTS.md for third-party model licenses.

import os
import sys
import glob
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models

# Paths
GVHMR_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
THREEDPW_DIR = os.path.join(GVHMR_ROOT, "inputs/3DPW")
IMGFEAT_DIR = os.path.join(THREEDPW_DIR, "hmr4d_support/imgfeats/3dpw_test")
IMAGE_DIR = os.path.join(THREEDPW_DIR, "imageFiles")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

# Also try train features
IMGFEAT_TRAIN_DIR = os.path.join(THREEDPW_DIR, "hmr4d_support/imgfeats/3dpw_train_sam3d")

# HMR2 preprocessing constants
IMAGE_MEAN = np.array([0.485, 0.456, 0.406])
IMAGE_STD = np.array([0.229, 0.224, 0.225])


# =============================================================================
# Mobile Feature Extractor Model
# =============================================================================
class MobileFeatureExtractor(nn.Module):
    """
    MobileNetV3-Small backbone with an adapter head that outputs 1024-dim
    features matching HMR2's token_out space.

    Input: (B, 3, 224, 224) RGB image, ImageNet-normalized
    Output: (B, 1024) feature vector
    """

    def __init__(self):
        super().__init__()
        # Use pretrained MobileNetV3-Small backbone
        backbone = models.mobilenet_v3_small(weights="DEFAULT")
        # Extract feature layers (everything except classifier)
        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # Adapter: backbone output (576-dim) → 1024-dim matching HMR2
        self.adapter = nn.Sequential(
            nn.Linear(576, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)  # (B, 576)
        x = self.adapter(x)  # (B, 1024)
        return x


# =============================================================================
# Dataset
# =============================================================================
class FeatureDistillDataset(Dataset):
    """
    Loads (image_crop, hmr2_feature) pairs from 3DPW.

    Each .pt file contains:
      - features: (F, 1024) HMR2 token_out features
      - bbx_xys: (F, 3) bounding box [cx, cy, size]
      - img_wh: (W, H)

    We crop images using bbx_xys to match what HMR2 saw.
    """

    def __init__(self, imgfeat_dirs, image_dir, max_samples=None):
        self.image_dir = image_dir
        self.samples = []  # (video_name, frame_idx, feature_1024, bbx_xy, bbx_s)

        for imgfeat_dir in imgfeat_dirs:
            if not os.path.isdir(imgfeat_dir):
                print(f"Skipping missing dir: {imgfeat_dir}")
                continue

            pt_files = sorted(glob.glob(os.path.join(imgfeat_dir, "*.pt")))
            for pt_file in pt_files:
                vid_name = os.path.basename(pt_file).replace(".pt", "")
                # Parse video name: e.g., "courtyard_golf_00_0" → seq="courtyard_golf_00"
                parts = vid_name.rsplit("_", 1)
                if len(parts) == 2:
                    seq_name = parts[0]
                else:
                    seq_name = vid_name

                img_seq_dir = os.path.join(image_dir, seq_name)
                if not os.path.isdir(img_seq_dir):
                    continue

                data = torch.load(pt_file, map_location="cpu")
                features = data["features"].float()  # (F, 1024)
                bbx_xys = data["bbx_xys"].float()  # (F, 3)
                n_frames = features.shape[0]

                # List available images
                img_files = sorted(glob.glob(os.path.join(img_seq_dir, "image_*.jpg")))
                n_images = len(img_files)

                # Only use frames that have both features and images
                for i in range(min(n_frames, n_images)):
                    img_path = os.path.join(img_seq_dir, f"image_{i:05d}.jpg")
                    if os.path.exists(img_path):
                        self.samples.append((
                            img_path,
                            features[i].numpy(),
                            bbx_xys[i].numpy(),
                        ))

        if max_samples and len(self.samples) > max_samples:
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

        print(f"FeatureDistillDataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, feature, bbx_xys = self.samples[idx]

        # Load and crop image
        img = cv2.imread(img_path)
        if img is None:
            # Return zeros if image can't be loaded
            return torch.zeros(3, 224, 224), torch.from_numpy(feature)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Crop around bbox (matching HMR2 preprocessing)
        cx, cy, s = bbx_xys[0], bbx_xys[1], bbx_xys[2]
        crop = self._crop_and_resize(img, cx, cy, s, dst_size=224)

        # Normalize (ImageNet)
        crop = crop.astype(np.float32) / 255.0
        crop = (crop - IMAGE_MEAN) / IMAGE_STD
        crop = torch.from_numpy(crop).permute(2, 0, 1).float()  # (3, 224, 224)

        return crop, torch.from_numpy(feature)

    def _crop_and_resize(self, img, cx, cy, s, dst_size=224, enlarge_ratio=1.2):
        """Crop and resize to square, matching HMR2's preprocessing."""
        hs = s * enlarge_ratio / 2
        bbx_xy = np.array([cx, cy])
        src = np.stack([
            bbx_xy - hs,
            bbx_xy + np.array([hs, -hs]),
            bbx_xy,
        ]).astype(np.float32)
        dst = np.array([
            [0, 0],
            [dst_size - 1, 0],
            [dst_size / 2 - 0.5, dst_size / 2 - 0.5],
        ], dtype=np.float32)
        A = cv2.getAffineTransform(src, dst)
        crop = cv2.warpAffine(img, A, (dst_size, dst_size), flags=cv2.INTER_LINEAR)
        return crop


# =============================================================================
# Training
# =============================================================================
def train():
    os.makedirs(CKPT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Dataset
    imgfeat_dirs = [IMGFEAT_DIR]
    if os.path.isdir(IMGFEAT_TRAIN_DIR):
        imgfeat_dirs.append(IMGFEAT_TRAIN_DIR)

    dataset = FeatureDistillDataset(
        imgfeat_dirs=imgfeat_dirs,
        image_dir=IMAGE_DIR,
        max_samples=50000,
    )

    if len(dataset) == 0:
        print("ERROR: No training samples found. Check paths.")
        return

    # Split train/val
    n_val = min(500, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Model
    model = MobileFeatureExtractor().to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(30):
        # Train
        model.train()
        train_loss = 0
        n_batches = 0
        for imgs, targets in train_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            pred = model(imgs)

            # Cosine similarity loss + MSE (both important for feature matching)
            cos_sim = F.cosine_similarity(pred, targets, dim=1).mean()
            mse = F.mse_loss(pred, targets)
            loss = mse + (1 - cos_sim)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train = train_loss / max(n_batches, 1)

        # Validate
        model.eval()
        val_loss = 0
        val_cos_sim = 0
        n_val_batches = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                pred = model(imgs)
                mse = F.mse_loss(pred, targets)
                cos_sim = F.cosine_similarity(pred, targets, dim=1).mean()
                val_loss += (mse + (1 - cos_sim)).item()
                val_cos_sim += cos_sim.item()
                n_val_batches += 1

        avg_val = val_loss / max(n_val_batches, 1)
        avg_cos = val_cos_sim / max(n_val_batches, 1)

        print(f"Epoch {epoch+1:3d} | train_loss={avg_train:.4f} | val_loss={avg_val:.4f} | cos_sim={avg_cos:.4f} | lr={scheduler.get_last_lr()[0]:.6f}")

        # Save best
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "val_loss": avg_val,
                "cos_sim": avg_cos,
            }, os.path.join(CKPT_DIR, "mobilenet_distilled.pt"))
            print(f"  → Saved best model (val_loss={avg_val:.4f}, cos_sim={avg_cos:.4f})")

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {os.path.join(CKPT_DIR, 'mobilenet_distilled.pt')}")


if __name__ == "__main__":
    train()
