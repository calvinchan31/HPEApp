# Place CoreML models here

After running `export_coreml.py`, copy these files into this directory:

- `MobileNetProxy.mlpackage` — Lightweight feature extractor
- `GVHMRStudent.mlpackage` — Distilled GVHMR transformer

Optional (for visual-language captioning):

- `GVHMRCaption.mlpackage` or
- `Captioner.mlpackage` or
- `MobileCaption.mlpackage`

Caption model interface expected by the app:

- Input feature: `image` (CVPixelBuffer)
- Output feature: string caption (preferably named `caption`)

Then add them to your Xcode project.
