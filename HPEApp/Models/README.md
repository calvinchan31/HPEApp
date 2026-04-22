# CoreML Models Directory

This folder is where the iOS app loads CoreML models from at runtime.

## Exported by `export_coreml.py`

- `MobileNetProxy.mlpackage` — Feature extractor (required)
- `GVHMRStudent.mlpackage` — Small GVHMR variant
- `GVHMRMedium.mlpackage` — Medium GVHMR variant
- `GVHMROriginal.mlpackage` — Original GVHMR variant

At least one GVHMR variant (`GVHMRStudent`, `GVHMRMedium`, or `GVHMROriginal`) must be present for inference.

## Also used by the app

- `SMPLForward.mlpackage` — SMPL mesh forward model (needed for mesh rendering)
- `ViTPoseSmall.mlpackage` — Needed for `Vision+ViTPose` and `YOLO+ViTPose` modes
- `yolo26n-pose.mlpackage` — Needed for `YOLO+ViTPose` and `YOLO-Pose` modes

## Optional caption model

The app will auto-detect the first available model from:

- `GVHMRCaption.mlpackage`
- `Captioner.mlpackage`
- `MobileCaption.mlpackage`

Expected caption model interface:

- Input feature: `image` (CVPixelBuffer)
- Output feature: string caption (preferably named `caption`)

If no caption model is present, the app falls back to pose-based semantic captions.
