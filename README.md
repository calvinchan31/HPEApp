# HPE App

Real-time human motion recovery on iPhone using selectable GVHMR model variants.

## Architecture

```
Camera (720p) ──► Preprocessing
                    (Vision+ViTPose OR YOLO+ViTPose OR YOLO-Pose)
                        │
Device Gyroscope ──► MotionManager (6D angular velocity)
                        │
Camera Intrinsics + BBox ──► CLIFF Camera features
                        │
Cropped Person ──► MobileNetV3-Small Proxy (1024D features)
                        │
    All inputs ──► FrameBuffer (16-frame sliding window)
                        │
       Selected GVHMR CoreML (Small / Medium / Original)
                        │
         SMPLDecoder (forward kinematics → 22 joints)
                        │
   CaptionFusionEngine (+ optional CoreMLCaptioner)
                        │
   SkeletonOverlayView / Mesh3DView (live + video views)
```

**Model Variants**:
- Small: 256-dim latent, 6 transformer layers, 4 attention heads
- Medium: 384-dim latent, 8 transformer layers, 6 attention heads
- Original: 512-dim latent, 12 transformer layers, 8 attention heads
- Temporal window: 16 frames, inference every 4 frames

## Prerequisites

- **macOS** with Xcode 15+ (iOS 16.0 SDK)
- **iPhone** with A12 Bionic or newer (arm64, iOS 16+)
- **XcodeGen**: `brew install xcodegen`
- **Python 3.8+** with PyTorch, coremltools, and GVHMR environment

## Setup

### 1. Export CoreML Models

From this folder (with the GVHMR conda env activated):

```bash
python export_coreml.py
```

Note: `export_coreml.py` expects GVHMR checkpoints/imports to be available via its configured
`ROOT` and checkpoint paths. If your local layout differs, adjust paths in `export_coreml.py`.

This produces:
- `HPEApp/Models/MobileNetProxy.mlpackage`
- `HPEApp/Models/GVHMRStudent.mlpackage`
- `HPEApp/Models/GVHMRMedium.mlpackage`
- `HPEApp/Models/GVHMROriginal.mlpackage`
- `HPEApp/gvhmr_stats.json` (normalization statistics)

### 2. Generate Xcode Project

```bash
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

- `export_coreml.py` — CoreML model export script
- `project.yml` — XcodeGen project definition
- `HPEApp/HPEApp.swift` — app entry point
- `HPEApp/ContentView.swift` — main SwiftUI shell (Live/Video modes)
- `HPEApp/GVHMRPipeline.swift` — live camera inference pipeline
- `HPEApp/VideoProcessingView.swift` — offline video processing and model comparison UI
- `HPEApp/Models/` — CoreML models bundled with the app
- `HPEApp/gvhmr_stats.json` — normalization statistics used by decoder

## Usage

1. Launch the app on your iPhone
2. Point the camera at a person
3. Tap **Start** to begin motion capture
4. Toggle **2D** to see detected keypoints (cyan)
5. Toggle **Mesh** to see 3D mesh view (or disable it for skeleton view)
6. Toggle **Caption** to show semantic action captions from motion output
7. In **Multi** mode, tap a person in the camera view to select them for Caption description
8. The Caption panel shows both person count and selected track id
9. FPS is displayed in the top-right corner

### Video Mode Controls

- **Caption toggle** in playback controls enables/disables semantic captions in video mode.
- In multi-person video mode, per-frame person chips (`P<track_id>`) let you choose which person's Caption timeline is shown.
- **Compare All Models** runs Small/Medium/Original and can show per-model preview panels.

### Live Stats Modes

- **Stats Off**: hide live diagnostics
- **Stats All**: full diagnostics (all metrics)
- **Stats Core**: compact diagnostics (GVHMR, Pipeline, Memory focus)

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
3. **SMPL**: See https://github.com/vchoutas/smplx/tree/main for citations

---

## Contributing

Contributions are welcome! By contributing, you agree to license your contributions under the MIT License (for app code).

---

## Support & Issues

For issues, feature requests, or questions:
1. Open an issue on GitHub
2. Review [ACKNOWLEDGMENTS.md](ACKNOWLEDGMENTS.md) for licensing questions
3. See [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md) for contribution and setup guidance
