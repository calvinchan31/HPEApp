import CoreML
import UIKit
import Accelerate
import CoreImage

/// Runs ViTPose-Small CoreML model to extract COCO-17 keypoints from a person crop.
/// Replaces Apple Vision keypoints with ViTPose keypoints that match the GVHMR training data.
///
/// Pipeline:
/// 1. Crop person from frame using bbox (square crop with 1.2x enlarge)
/// 2. Resize to 256×192 (matching ViTPose input)
/// 3. Normalize with ImageNet stats
/// 4. Run ViTPose → heatmaps (17, 64, 48)
/// 5. Decode heatmaps to pixel coordinates via argmax + subpixel refinement
class VitPoseProcessor {

    private var model: MLModel?

    /// ImageNet normalization constants
    private let means: [Float] = [0.485, 0.456, 0.406]
    private let stds: [Float] = [0.229, 0.224, 0.225]

    var isReady: Bool { model != nil }

    // MARK: - Model Loading

    func loadModel() {
        let extensions = ["mlmodelc", "mlpackage"]
        for ext in extensions {
            if let url = Bundle.main.url(forResource: "ViTPoseSmall", withExtension: ext) {
                do {
                    let config = MLModelConfiguration()
                    config.computeUnits = .cpuAndNeuralEngine
                    model = try MLModel(contentsOf: url, configuration: config)
                    print("[ViTPose] Loaded ViTPoseSmall.\(ext)")
                    return
                } catch {
                    print("[ViTPose] Error loading \(ext): \(error)")
                }
            }
        }
        print("[ViTPose] Model not found in bundle")
    }

    // MARK: - Full Pipeline

    /// Extract COCO-17 keypoints from a video frame given person bounding box.
    /// - Parameters:
    ///   - pixelBuffer: Full video frame (BGRA)
    ///   - bbxXYS: Bounding box as (center_x, center_y, size) with 1.2x enlarge + aspect ratio
    ///   - imageSize: Frame dimensions
    /// - Returns: 17 keypoints as (x_pixel, y_pixel, confidence), or nil
    func extractKeypoints(pixelBuffer: CVPixelBuffer, bbxXYS: SIMD3<Float>, imageSize: CGSize) -> [SIMD3<Float>]? {
        guard let model = model else { return nil }

        // 1. Crop and resize to (256, 192)
        // Python pipeline: crop to square (bbx_size × bbx_size), resize to 256×256,
        // then slice columns [32:224] to get 256×192.
        // We can equivalently crop a 3:4 aspect ratio region and resize to 192×256 directly.
        let cx = CGFloat(bbxXYS.x)
        let cy = CGFloat(bbxXYS.y)
        let bbxSize = CGFloat(bbxXYS.z)

        // The ViTPose input is 256h × 192w. The crop should maintain 3:4 aspect ratio.
        // Python: crops square, resizes to 256×256, then takes columns [32:224] (width 192).
        // Equivalent: crop width = bbxSize * 192/256, height = bbxSize
        let cropH = bbxSize
        let cropW = bbxSize * 192.0 / 256.0

        guard let inputArray = cropAndPrepare(
            pixelBuffer: pixelBuffer,
            centerX: cx, centerY: cy,
            cropWidth: cropW, cropHeight: cropH,
            targetWidth: 192, targetHeight: 256,
            imageSize: imageSize
        ) else {
            return nil
        }

        // 2. Run ViTPose model
        do {
            let inputProvider = try MLDictionaryFeatureProvider(dictionary: ["image": inputArray])
            let output = try model.prediction(from: inputProvider)

            // 3. Get heatmaps output: (1, 17, 64, 48)
            guard let heatmaps = findHeatmapOutput(output) else {
                print("[ViTPose] Could not find heatmap output")
                return nil
            }

            // 4. Decode heatmaps to pixel coordinates
            return decodeHeatmaps(heatmaps, bbxXYS: bbxXYS)

        } catch {
            print("[ViTPose] Inference error: \(error)")
            return nil
        }
    }

    // MARK: - Image Preprocessing

    /// Shared CIContext for hardware-accelerated image operations.
    private lazy var ciContext = CIContext(options: [.useSoftwareRenderer: false])

    /// Crop a region from the pixel buffer, resize to target size, and normalize.
    /// Uses CIImage for crop+resize (GPU-accelerated) then reads into MLMultiArray.
    private func cropAndPrepare(
        pixelBuffer: CVPixelBuffer,
        centerX: CGFloat, centerY: CGFloat,
        cropWidth: CGFloat, cropHeight: CGFloat,
        targetWidth: Int, targetHeight: Int,
        imageSize: CGSize
    ) -> MLMultiArray? {
        let srcWidth = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let srcHeight = CGFloat(CVPixelBufferGetHeight(pixelBuffer))

        // Crop region in source coordinates — clamp to image bounds
        let cropX = max(0, centerX - cropWidth / 2)
        let cropY = max(0, centerY - cropHeight / 2)
        let cropR = min(srcWidth, centerX + cropWidth / 2)
        let cropB = min(srcHeight, centerY + cropHeight / 2)
        let clampedW = cropR - cropX
        let clampedH = cropB - cropY

        guard clampedW > 0, clampedH > 0 else { return nil }

        // CIImage has origin at bottom-left; convert top-left y to bottom-left
        let ciCropRect = CGRect(
            x: cropX,
            y: srcHeight - cropB,
            width: clampedW,
            height: clampedH
        )

        // CIImage crop + resize
        var ci = CIImage(cvPixelBuffer: pixelBuffer)
        ci = ci.cropped(to: ciCropRect)
        ci = ci.transformed(by: CGAffineTransform(
            translationX: -ciCropRect.origin.x,
            y: -ciCropRect.origin.y
        ))

        let sx = CGFloat(targetWidth) / clampedW
        let sy = CGFloat(targetHeight) / clampedH
        ci = ci.transformed(by: CGAffineTransform(scaleX: sx, y: sy))

        // Render to a temporary BGRA buffer
        let bpr = targetWidth * 4
        var bgraBuf = [UInt8](repeating: 0, count: targetHeight * bpr)
        let renderRect = CGRect(x: 0, y: 0, width: targetWidth, height: targetHeight)
        ciContext.render(ci, toBitmap: &bgraBuf, rowBytes: bpr,
                         bounds: renderRect, format: .BGRA8, colorSpace: nil)

        // Build MLMultiArray (1, 3, H, W) with ImageNet normalization
        guard let array = try? MLMultiArray(
            shape: [1, 3, NSNumber(value: targetHeight), NSNumber(value: targetWidth)],
            dataType: .float32
        ) else { return nil }

        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: 3 * targetHeight * targetWidth)
        let count = targetHeight * targetWidth

        for i in 0..<count {
            let off = i * 4
            // BGRA → RGB, normalize [0,1], ImageNet (x - mean) / std
            ptr[0 * count + i] = (Float(bgraBuf[off + 2]) / 255.0 - means[0]) / stds[0] // R
            ptr[1 * count + i] = (Float(bgraBuf[off + 1]) / 255.0 - means[1]) / stds[1] // G
            ptr[2 * count + i] = (Float(bgraBuf[off + 0]) / 255.0 - means[2]) / stds[2] // B
        }

        return array
    }

    // MARK: - Heatmap Decoding

    /// Find the heatmap output tensor from model output.
    private func findHeatmapOutput(_ output: MLFeatureProvider) -> MLMultiArray? {
        for name in output.featureNames.sorted() {
            if let arr = output.featureValue(for: name)?.multiArrayValue {
                let shape = arr.shape.map { $0.intValue }
                // Expect (1, 17, 64, 48)
                if shape.count == 4 && shape[1] == 17 {
                    return arr
                }
            }
        }
        return nil
    }

    /// Decode heatmaps (1, 17, 64, 48) to pixel coordinates.
    /// Uses argmax + subpixel refinement, then transforms back to image coordinates.
    /// Matches mmpose's keypoints_from_heatmaps with use_udp=True.
    private func decodeHeatmaps(_ heatmaps: MLMultiArray, bbxXYS: SIMD3<Float>) -> [SIMD3<Float>] {
        let K = 17   // joints
        let H = 64   // heatmap height
        let W = 48   // heatmap width
        let total = K * H * W

        // CoreML FLOAT16 models return Float16 MLMultiArrays.
        // Must convert to Float32 before pointer access.
        var heatmapData = [Float](repeating: 0, count: total)
        if heatmaps.dataType == .float32 {
            let rawPtr = heatmaps.dataPointer.bindMemory(to: Float.self, capacity: total)
            for i in 0..<total { heatmapData[i] = rawPtr[i] }
        } else {
            for i in 0..<total { heatmapData[i] = heatmaps[i].floatValue }
        }

        var keypoints = [SIMD3<Float>]()
        keypoints.reserveCapacity(K)

        // bbx_xys: (cx, cy, size)
        // In Python: scale = (bbx_xys[2] * 24/32, bbx_xys[2]) / 200
        // transform_preds: scale * 200, then scale_x = scale[0] / (W-1), scale_y = scale[1] / (H-1)
        let bbxSize = bbxXYS.z
        let scaleW = bbxSize * (24.0 / 32.0)  // width scale
        let scaleH = bbxSize                    // height scale

        // UDP: scale_x = scaleW / (W-1), scale_y = scaleH / (H-1)
        let scaleX = scaleW / Float(W - 1)
        let scaleY = scaleH / Float(H - 1)

        // Origin offset: center - scale/2
        let originX = bbxXYS.x - scaleW * 0.5
        let originY = bbxXYS.y - scaleH * 0.5

        for j in 0..<K {
            // Find argmax in heatmap[j]
            var maxVal: Float = -Float.greatestFiniteMagnitude
            var maxIdx = 0
            let baseOffset = j * H * W

            for i in 0..<(H * W) {
                let val = heatmapData[baseOffset + i]
                if val > maxVal {
                    maxVal = val
                    maxIdx = i
                }
            }

            var predX = Float(maxIdx % W)
            var predY = Float(maxIdx / W)

            // Subpixel refinement (DARK UDP post-processing simplified)
            // Uses gradient-based shift for sub-pixel accuracy
            let ix = maxIdx % W
            let iy = maxIdx / W
            if ix > 1 && ix < W - 2 && iy > 1 && iy < H - 2 {
                let left = heatmapData[baseOffset + iy * W + ix - 1]
                let right = heatmapData[baseOffset + iy * W + ix + 1]
                let top = heatmapData[baseOffset + (iy - 1) * W + ix]
                let bottom = heatmapData[baseOffset + (iy + 1) * W + ix]

                // Gradient
                let dx = 0.5 * (right - left)
                let dy = 0.5 * (bottom - top)

                // Hessian
                let center = heatmapData[baseOffset + iy * W + ix]
                let dxx = right - 2 * center + left
                let dyy = bottom - 2 * center + top

                let topLeft = heatmapData[baseOffset + (iy - 1) * W + ix - 1]
                let topRight = heatmapData[baseOffset + (iy - 1) * W + ix + 1]
                let botLeft = heatmapData[baseOffset + (iy + 1) * W + ix - 1]
                let botRight = heatmapData[baseOffset + (iy + 1) * W + ix + 1]
                let dxy = 0.25 * (botRight - botLeft - topRight + topLeft)

                let det = dxx * dyy - dxy * dxy
                if abs(det) > 1e-6 {
                    // Newton step: offset = -H^{-1} * g
                    let invDet = 1.0 / det
                    predX -= invDet * (dyy * dx - dxy * dy)
                    predY -= invDet * (-dxy * dx + dxx * dy)
                }
            }

            // Transform to image coordinates (UDP)
            let pixelX = predX * scaleX + originX
            let pixelY = predY * scaleY + originY

            keypoints.append(SIMD3<Float>(pixelX, pixelY, maxVal))
        }

        return keypoints
    }
}
