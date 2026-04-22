import CoreML
import CoreGraphics
import UIKit

/// Runs YOLO26n-Pose CoreML model to detect a person and extract COCO-17 keypoints in a single pass.
/// Replaces the two-step Apple Vision (bbox) + ViTPose (keypoints) pipeline.
///
/// IMPORTANT: YOLO models expect **letterboxed** input (uniform scale + gray padding),
/// NOT stretch-resized input. CoreML's imageType does a stretch-resize by default,
/// which destroys aspect ratio and kills detection quality. We must letterbox manually.
///
/// Output format (1, 300, 57) per detection (in letterbox-640 coordinate space):
///   [0:4]  = x1, y1, x2, y2 (bbox corners)
///   [4]    = confidence score
///   [5]    = class_id (always 0 for person)
///   [6:57] = 17 × (x, y, visibility) keypoints
class YOLOPoseProcessor {

    private var model: MLModel?
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    /// Input image size expected by the model.
    private let inputSize: Int = 640

    /// Minimum confidence threshold for accepting a detection.
    private let confidenceThreshold: Float = 0.25

    /// Letterbox padding fill color (matches ultralytics default: 114/255).
    private let padGray: CGFloat = 114.0 / 255.0

    // MARK: - Temporal Tracking State (mirrors PoseDetector)

    private var previousBBox: CGRect?
    private var holdFrameCount: Int = 0
    private let maxHoldFrames = 15

    var isReady: Bool { model != nil }

    // MARK: - Model Loading

    func loadModel() {
        let extensions = ["mlmodelc", "mlpackage"]
        for ext in extensions {
            if let url = Bundle.main.url(forResource: "yolo26n-pose", withExtension: ext) {
                do {
                    let config = MLModelConfiguration()
                    config.computeUnits = .cpuAndNeuralEngine
                    model = try MLModel(contentsOf: url, configuration: config)
                    print("[YOLOPose] Loaded yolo26n-pose.\(ext)")
                    return
                } catch {
                    print("[YOLOPose] Error loading \(ext): \(error)")
                }
            }
        }
        print("[YOLOPose] Model not found in bundle")
    }

    func resetTracking() {
        previousBBox = nil
        holdFrameCount = 0
    }

    // MARK: - Letterbox Preprocessing

    /// Letterbox-resize an image to 640×640 maintaining aspect ratio with gray padding.
    /// Returns the letterboxed CGImage plus the transform parameters needed to undo it.
    private func letterbox(cgImage: CGImage) -> (image: CGImage, scale: Float, padW: Float, padH: Float)? {
        let srcW = cgImage.width
        let srcH = cgImage.height
        let sz = inputSize

        // Uniform scale to fit inside 640×640
        let scale = min(Float(sz) / Float(srcW), Float(sz) / Float(srcH))
        let newW = Int(round(Float(srcW) * scale))
        let newH = Int(round(Float(srcH) * scale))

        // Padding to center the resized image
        let padW = Float(sz - newW) / 2.0
        let padH = Float(sz - newH) / 2.0
        let left = Int(floor(padW))
        let top  = Int(floor(padH))

        // Create 640×640 context with gray fill (114, 114, 114)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let ctx = CGContext(
            data: nil,
            width: sz, height: sz,
            bitsPerComponent: 8, bytesPerRow: sz * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
        ) else { return nil }

        // Fill with letterbox gray
        ctx.setFillColor(CGColor(red: padGray, green: padGray, blue: padGray, alpha: 1.0))
        ctx.fill(CGRect(x: 0, y: 0, width: sz, height: sz))

        // Draw resized image centered.
        // CGContext has origin at bottom-left, so flip the y offset.
        let drawRect = CGRect(x: left, y: sz - top - newH, width: newW, height: newH)
        ctx.interpolationQuality = .high
        ctx.draw(cgImage, in: drawRect)

        guard let letterboxed = ctx.makeImage() else { return nil }
        return (letterboxed, scale, padW, padH)
    }

    // MARK: - Detection

    /// Detect person and extract 17 COCO keypoints in one pass.
    /// - Parameters:
    ///   - pixelBuffer: Full video frame (BGRA)
    ///   - imageSize: Frame dimensions in pixels
    /// - Returns: (keypoints in pixel coords, bounding box in pixels), or nil if no person found
    func detect(pixelBuffer: CVPixelBuffer, imageSize: CGSize) -> (keypoints: [SIMD3<Float>], bbox: CGRect)? {
        guard let model = model else { return nil }

        // Convert CVPixelBuffer → CGImage
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else { return nil }

        // Letterbox to 640×640 (maintain aspect ratio, gray padding)
        guard let lb = letterbox(cgImage: cgImage) else { return nil }

        do {
            // Pass letterboxed 640×640 image to CoreML
            // Since it's already 640×640, CoreML's imageType resize is a no-op
            guard let imageConstraint = model.modelDescription.inputDescriptionsByName["image"]?.imageConstraint else {
                print("[YOLOPose] Model has no image input constraint")
                return nil
            }
            let imageFeature = try MLFeatureValue(cgImage: lb.image, constraint: imageConstraint)
            let input = try MLDictionaryFeatureProvider(dictionary: ["image": imageFeature])
            let output = try model.prediction(from: input)

            // Find the output tensor (shape 1×300×57)
            guard let rawOutput = findOutputTensor(output) else {
                print("[YOLOPose] Could not find output tensor")
                return nil
            }

            return parseBestDetection(rawOutput, scale: lb.scale, padW: lb.padW, padH: lb.padH)
        } catch {
            print("[YOLOPose] Inference error: \(error)")
            return nil
        }
    }

    // MARK: - Output Parsing

    /// Find the output tensor from model output (shape 1×300×57).
    private func findOutputTensor(_ output: MLFeatureProvider) -> MLMultiArray? {
        for name in output.featureNames {
            if let arr = output.featureValue(for: name)?.multiArrayValue {
                let shape = arr.shape.map { $0.intValue }
                if shape.count == 3 && shape[2] == 57 {
                    return arr
                }
            }
        }
        return nil
    }

    private struct Detection {
        let confidence: Float
        let bbox: CGRect
        let keypoints: [SIMD3<Float>]
    }

    /// Parse model output and return the best-tracked detection.
    /// Coordinates in the output are in 640×640 letterbox space.
    /// We undo the letterbox: x_orig = (x_640 - padW) / scale
    ///
    /// IMPORTANT: CoreML MLMultiArray does NOT guarantee row-major (C-contiguous) memory layout.
    /// We MUST use the strides property to correctly index into the raw data pointer.
    private func parseBestDetection(_ output: MLMultiArray, scale: Float, padW: Float, padH: Float) -> (keypoints: [SIMD3<Float>], bbox: CGRect)? {
        let numDetections = output.shape[1].intValue
        let detDim = 57

        var detections = [Detection]()

        // Read data using stride-aware indexing.
        // Shape is (1, 300, 57). Strides tell us the memory offset per index increment.
        let strides = output.strides.map { $0.intValue }
        let s1 = strides[1]  // stride along detection axis
        let s2 = strides[2]  // stride along feature axis

        // Copy into row-major buffer for easy downstream access
        var data = [Float](repeating: 0, count: numDetections * detDim)
        if output.dataType == .float32 {
            let ptr = output.dataPointer.bindMemory(to: Float.self, capacity: output.count)
            for i in 0..<numDetections {
                let rowBase = i * detDim
                let memBase = i * s1
                for j in 0..<detDim {
                    data[rowBase + j] = ptr[memBase + j * s2]
                }
            }
        } else {
            for i in 0..<numDetections {
                let rowBase = i * detDim
                for j in 0..<detDim {
                    data[rowBase + j] = output[[0, i, j] as [NSNumber]].floatValue
                }
            }
        }

        for i in 0..<numDetections {
            let base = i * detDim
            let conf = data[base + 4]
            if conf < confidenceThreshold { continue }

            // Bbox: x1, y1, x2, y2 in letterbox-640 space → original image
            let x1 = (data[base + 0] - padW) / scale
            let y1 = (data[base + 1] - padH) / scale
            let x2 = (data[base + 2] - padW) / scale
            let y2 = (data[base + 3] - padH) / scale

            let bbox = CGRect(
                x: CGFloat(x1), y: CGFloat(y1),
                width: CGFloat(x2 - x1), height: CGFloat(y2 - y1)
            )

            // 17 keypoints: (x, y, visibility) starting at index 6
            var kps = [SIMD3<Float>]()
            kps.reserveCapacity(17)
            for j in 0..<17 {
                let kpBase = base + 6 + j * 3
                let kpx = (data[kpBase + 0] - padW) / scale
                let kpy = (data[kpBase + 1] - padH) / scale
                let kpc = data[kpBase + 2]
                kps.append(SIMD3<Float>(kpx, kpy, kpc))
            }

            detections.append(Detection(confidence: conf, bbox: bbox, keypoints: kps))
        }

        guard !detections.isEmpty else { return nil }

        // IoU-based temporal tracking
        let chosen: Detection
        if let prev = previousBBox {
            var bestIoU: Float = -1
            var bestDet = detections[0]
            for det in detections {
                let iou = computeIoU(prev, det.bbox)
                if iou > bestIoU {
                    bestIoU = iou
                    bestDet = det
                }
            }

            if bestIoU >= 0.2 {
                holdFrameCount = 0
                chosen = bestDet
            } else {
                holdFrameCount += 1
                if holdFrameCount <= maxHoldFrames {
                    chosen = detections.max(by: { $0.confidence < $1.confidence })!
                } else {
                    holdFrameCount = 0
                    chosen = detections.max(by: { $0.confidence < $1.confidence })!
                }
            }
        } else {
            chosen = detections.max(by: { $0.confidence < $1.confidence })!
            holdFrameCount = 0
        }

        previousBBox = chosen.bbox
        return (chosen.keypoints, chosen.bbox)
    }

    private func computeIoU(_ a: CGRect, _ b: CGRect) -> Float {
        let intersection = a.intersection(b)
        if intersection.isNull { return 0 }
        let interArea = intersection.width * intersection.height
        let unionArea = a.width * a.height + b.width * b.height - interArea
        guard unionArea > 0 else { return 0 }
        return Float(interArea / unionArea)
    }

    // MARK: - Multi-Person Detection

    /// Detect ALL persons above confidence threshold. Returns detections sorted by bbox area (largest first).
    /// No tracking is applied — each frame is independent.
    func detectAll(pixelBuffer: CVPixelBuffer, imageSize: CGSize, maxPersons: Int = 5) -> [(keypoints: [SIMD3<Float>], bbox: CGRect, confidence: Float)] {
        guard let model = model else { return [] }

        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else { return [] }
        guard let lb = letterbox(cgImage: cgImage) else { return [] }

        do {
            guard let imageConstraint = model.modelDescription.inputDescriptionsByName["image"]?.imageConstraint else { return [] }
            let imageFeature = try MLFeatureValue(cgImage: lb.image, constraint: imageConstraint)
            let input = try MLDictionaryFeatureProvider(dictionary: ["image": imageFeature])
            let output = try model.prediction(from: input)
            guard let rawOutput = findOutputTensor(output) else { return [] }

            return parseAllDetections(rawOutput, scale: lb.scale, padW: lb.padW, padH: lb.padH, maxPersons: maxPersons)
        } catch {
            return []
        }
    }

    /// Parse model output and return ALL detections above threshold, sorted by bbox area.
    private func parseAllDetections(_ output: MLMultiArray, scale: Float, padW: Float, padH: Float, maxPersons: Int) -> [(keypoints: [SIMD3<Float>], bbox: CGRect, confidence: Float)] {
        let numDetections = output.shape[1].intValue
        let detDim = 57

        let strides = output.strides.map { $0.intValue }
        let s1 = strides[1]
        let s2 = strides[2]

        var data = [Float](repeating: 0, count: numDetections * detDim)
        if output.dataType == .float32 {
            let ptr = output.dataPointer.bindMemory(to: Float.self, capacity: output.count)
            for i in 0..<numDetections {
                let rowBase = i * detDim
                let memBase = i * s1
                for j in 0..<detDim {
                    data[rowBase + j] = ptr[memBase + j * s2]
                }
            }
        } else {
            for i in 0..<numDetections {
                let rowBase = i * detDim
                for j in 0..<detDim {
                    data[rowBase + j] = output[[0, i, j] as [NSNumber]].floatValue
                }
            }
        }

        var results = [(keypoints: [SIMD3<Float>], bbox: CGRect, confidence: Float)]()

        for i in 0..<numDetections {
            let base = i * detDim
            let conf = data[base + 4]
            if conf < confidenceThreshold { continue }

            let x1 = (data[base + 0] - padW) / scale
            let y1 = (data[base + 1] - padH) / scale
            let x2 = (data[base + 2] - padW) / scale
            let y2 = (data[base + 3] - padH) / scale

            let bbox = CGRect(x: CGFloat(x1), y: CGFloat(y1),
                              width: CGFloat(x2 - x1), height: CGFloat(y2 - y1))

            var kps = [SIMD3<Float>]()
            kps.reserveCapacity(17)
            for j in 0..<17 {
                let kpBase = base + 6 + j * 3
                let kpx = (data[kpBase + 0] - padW) / scale
                let kpy = (data[kpBase + 1] - padH) / scale
                let kpc = data[kpBase + 2]
                kps.append(SIMD3<Float>(kpx, kpy, kpc))
            }

            results.append((keypoints: kps, bbox: bbox, confidence: conf))
        }

        // Sort by bbox area (largest first), then take top maxPersons
        results.sort { $0.bbox.width * $0.bbox.height > $1.bbox.width * $1.bbox.height }

        // NMS: remove detections with high IoU to a larger detection
        var kept = [(keypoints: [SIMD3<Float>], bbox: CGRect, confidence: Float)]()
        for det in results {
            var dominated = false
            for existing in kept {
                if computeIoU(existing.bbox, det.bbox) > 0.5 {
                    dominated = true
                    break
                }
            }
            if !dominated {
                kept.append(det)
                if kept.count >= maxPersons { break }
            }
        }

        return kept
    }
}
