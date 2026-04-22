# HPE App

Real-time human motion recovery on iPhone using a distilled GVHMR student model.

## Architecture

```
Camera (720p) ──► Apple Vision (2D Pose + BBox)
                        │
Device Gyroscope ──► MotionManager (6D angular velocity)
                        │
Camera Intrinsics ──► CLIFF Camera (3D)
                        │
Cropped Person ──► MobileNetV3-Small Proxy (1024D features)
                        │
    All inputs ──► FrameBuffer (16-frame sliding window)
                        │
                   GVHMRStudent CoreML (pred_x: 151D, pred_cam: 3D)
                        │
                   SMPLDecoder (forward kinematics → 22 joints)
                        │
                  CaptionFusionEngine (semantic caption)
                        │
                   SkeletonOverlayView (Canvas rendering)
```

**Model**: Small student (~5.79M params) distilled from the full GVHMR teacher (~40.85M params).
- 256-dim latent, 6 transformer layers, 4 attention heads
- Temporal window: 16 frames, inference every 4 frames

## Prerequisites

- **macOS** with Xcode 15+ (iOS 16.0 SDK)
- **iPhone** with A12 Bionic or newer (arm64, iOS 16+)
- **XcodeGen**: `brew install xcodegen`
- **Python 3.8+** with PyTorch, coremltools, and GVHMR environment

## Setup

### 1. Export CoreML Models

From the GVHMR8 root (with the GVHMR conda env activated):

```bash
cd experiments/2026-02-22-iOSAPP
python export_coreml.py
```

This produces:
- `HPEApp/Models/MobileNetProxy.mlpackage`
- `HPEApp/Models/GVHMRStudent.mlpackage`
- `HPEApp/gvhmr_stats.json` (normalization statistics)

### 2. Generate Xcode Project

```bash
cd experiments/2026-02-22-iOSAPP
xcodegen generate
```

This creates `HPEApp.xcodeproj` from `project.yml`.

### Optional: Add CoreML Caption Model

The app now supports a real CoreML caption model for visual-language fusion.
Drop one of these model names into `HPEApp/Models/`:

- `GVHMRCaption.mlpackage`
- `Captioner.mlpackage`
- `MobileCaption.mlpackage`

Expected interface:
- Input feature: `image` (CVPixelBuffer)
- Output feature: a String caption (preferably named `caption`)

If no Caption model is present, the app falls back to pose-only semantic captions.

### Caption Semantic Outputs

The app Caption layer now outputs actionable semantic states, including:

- Estimated occluded parts (e.g., `left_hand`, `right_leg`)
- Raised limbs (`left_hand`, `right_hand`, `left_leg`, `right_leg`)
- Posture classes (`lying`, `bending`, `squatting`, `sitting`, `upright`)
- Motion/action classes (`running`, `walking`, `standing`, `hands_up`)

In multi-person mode, captions are computed per tracked person and the selected person
(tap on the camera overlay) is used for the displayed Caption description.

### 3. Open in Xcode

```bash
open HPEApp.xcodeproj
```

- Select your physical iPhone as the run target (camera doesn't work in simulator)
- Set your Team in Signing & Capabilities
- Build & Run (⌘R)

## Project Structure

```
experiments/2026-02-22-iOSAPP/
├── export_coreml.py          # CoreML model export script
├── project.yml               # XcodeGen project definition
├── README.md                 # This file
└── HPEApp/
     ├── HPEApp.swift          # App entry point
    ├── Types.swift           # Shared types, constants, enums
    ├── MathUtils.swift       # Rotation math (6D→matrix, Rodrigues, etc.)
    ├── FrameBuffer.swift     # 16-frame circular buffer
    ├── SMPLDecoder.swift     # Denormalize + forward kinematics (22 joints)
    ├── CameraManager.swift   # AVFoundation camera capture
    ├── MotionManager.swift   # CoreMotion gyroscope → angular velocity
    ├── PoseDetector.swift    # Apple Vision 2D pose + bounding box
    ├── GVHMRInference.swift  # CoreML model loading & inference
    ├── GVHMRPipeline.swift   # Main pipeline orchestrator
     ├── CaptionFusionEngine.swift # Pose-to-language semantic caption bridge
    ├── CameraPreviewView.swift   # UIKit camera preview wrapper
    ├── SkeletonOverlayView.swift # Canvas-based skeleton rendering
    ├── ContentView.swift     # Main SwiftUI UI
    ├── Info.plist            # Privacy permissions
    ├── gvhmr_stats.json      # Normalization mean/std (151 dims)
    ├── Models/               # CoreML .mlpackage files (after export)
    └── Assets.xcassets/      # App assets
```

## Usage

1. Launch the app on your iPhone
2. Point the camera at a person
3. Tap **Start** to begin motion capture
4. Toggle **2D Input** to see detected keypoints (cyan)
5. Toggle **3D Output** to see the SMPL skeleton overlay (color-coded)
6. Toggle **Caption** to show semantic action captions from motion output
7. In **Multi** mode, tap a person in the camera view to select them for Caption description
8. The Caption panel shows both person count and selected track id
9. FPS is displayed in the top-right corner

### Video Mode Controls

- **Caption toggle** in playback controls enables/disables semantic captions in video mode.
- In multi-person video mode, per-frame person chips (`P<track_id>`) let you choose which
     person's Caption timeline is shown.

### Live Stats Modes

- **Stats A**: full diagnostics (all metrics)
- **Stats C**: compact diagnostics (GVHMR, Pipeline, Memory focus)

## Technical Details

### Input Format
| Input | Shape | Description |
|-------|-------|-------------|
| obs | (1, 16, 17, 3) | COCO-17 keypoints (x, y, confidence), normalized to bbox |
| f_cliffcam | (1, 16, 3) | CLIFF camera: [(cx-icx), (cy-icy), bbox_size] / focal |
| f_cam_angvel | (1, 16, 6) | Inter-frame rotation (flattened 2×3 matrix) |
| f_imgseq | (1, 16, 1024) | MobileNetV3-Small image features |

### Output Format
| Output | Shape | Description |
|--------|-------|-------------|
| pred_x | (1, 16, 151) | Body params: pose_r6d(126) + betas(10) + orient_c(6) + orient_gv(6) + transl_vel(3) |
| pred_cam | (1, 16, 3) | Weak-perspective camera [s, tx, ty] |

### Skeleton Color Coding
- **Blue**: Left side (arm, leg)
- **Red**: Right side (arm, leg)
- **Green**: Spine / torso
- **Gray**: Other connections

---

## License

This iOS application code is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

### Third-Party Models & Attribution

This application incorporates research models and open-source components with their own licenses:

| Component | License | Usage |
|-----------|---------|-------|
| **GVHMR** | Research License | Core human motion recovery model |
| **ViTPose** | Apache 2.0 | 2D pose detection |
| **SMPL** | Research License | 3D body model |
| **YOLOv8** | AGPL-3.0 | Optional pose preprocessing |
| **PyTorch** | BSD 3-Clause | Deep learning framework |

 **Important**: Some models have commercial use restrictions. If you plan to commercialize this app, please review [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md) for licensing compliance requirements.

### Citation

If you use this app or code in research, please cite:

1. **GVHMR**: See https://github.com/zju3dv/GVHMR for citations
2. **ViTPose**: See https://github.com/ViTAE-Transformer/ViTPose for citations
3. **SMPL**: Loper et al., ACM TOG 2015

---

## Contributing

Contributions are welcome! By contributing, you agree to license your contributions under the MIT License (for app code).

---

## Support & Issues

For issues, feature requests, or questions:
1. Check [app_documentation_1.md](app_documentation_1.md) for detailed technical documentation
2. Open an issue on GitHub
3. Review [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md) for licensing questions
