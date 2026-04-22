// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Calvin Chan
// 
// CoreML model management for GVHMR and SMPL inference.
// Model licensing: See ACKNOWLEDGMENTS.md

import CoreML
import UIKit

/// Manages CoreML model loading and inference for both
/// the MobileNet feature extractor and GVHMR model variants (Small/Medium/Original).
class GVHMRInference {

    private var mobileNetModel: MLModel?
    private var gvhmrModels: [GVHMRModelChoice: MLModel] = [:]
    private var activeModelChoice: GVHMRModelChoice = .small
    private var smplModel: MLModel?
    private let queue = DispatchQueue(label: "com.gvhmr.inference", qos: .userInitiated)

    /// The currently active GVHMR model for inference.
    var activeGVHMRModel: MLModel? {
        return gvhmrModels[activeModelChoice]
    }

    /// Whether the active model is loaded and ready.
    var isReady: Bool {
        return mobileNetModel != nil && activeGVHMRModel != nil
    }

    /// Whether SMPL mesh model is available.
    var smplReady: Bool {
        return smplModel != nil
    }

    /// Which models are available in the bundle.
    var availableModels: [GVHMRModelChoice] {
        return GVHMRModelChoice.allCases.filter { gvhmrModels[$0] != nil }
    }

    /// Diagnostic info for debugging model loading issues.
    var diagnosticInfo: String = ""

    // MARK: - Model Loading

    func loadModels() {
        var diag = [String]()

        mobileNetModel = loadModel(named: "MobileNetProxy", diagnostics: &diag)

        // Load all available GVHMR model variants
        for choice in GVHMRModelChoice.allCases {
            if let model = loadModel(named: choice.modelName, diagnostics: &diag) {
                gvhmrModels[choice] = model
            }
        }

        smplModel = loadModel(named: "SMPLForward", diagnostics: &diag)

        // Default to first available model
        if gvhmrModels[activeModelChoice] == nil,
           let first = GVHMRModelChoice.allCases.first(where: { gvhmrModels[$0] != nil }) {
            activeModelChoice = first
        }

        diagnosticInfo = diag.joined(separator: "\n")
        print(diagnosticInfo)
    }

    /// Switch the active GVHMR model variant.
    func selectModel(_ choice: GVHMRModelChoice) {
        guard gvhmrModels[choice] != nil else {
            print("[GVHMRInference] Model \(choice.rawValue) not available")
            return
        }
        activeModelChoice = choice
        print("[GVHMRInference] Switched to \(choice.rawValue) (\(choice.detail))")
    }

    /// The currently selected model variant.
    var selectedModel: GVHMRModelChoice {
        return activeModelChoice
    }

    private func loadModel(named name: String, diagnostics: inout [String]) -> MLModel? {
        // Try compiled model first (.mlmodelc), then package (.mlpackage)
        let extensions = ["mlmodelc", "mlpackage"]
        for ext in extensions {
            if let url = Bundle.main.url(forResource: name, withExtension: ext) {
                do {
                    let config = MLModelConfiguration()
                    config.computeUnits = .cpuAndNeuralEngine
                    let model = try MLModel(contentsOf: url, configuration: config)
                    diagnostics.append("[OK] \(name).\(ext)")
                    let outputNames = model.modelDescription.outputDescriptionsByName.keys.sorted()
                    diagnostics.append("  outputs: \(outputNames.joined(separator: ", "))")
                    return model
                } catch {
                    diagnostics.append("[ERR] \(name).\(ext): \(error.localizedDescription)")
                }
            }
        }
        diagnostics.append("[MISS] \(name) not found in bundle")
        return nil
    }

    // MARK: - MobileNet Feature Extraction

    /// Extract 1024-dim image features from a cropped person image.
    func extractFeatures(pixelBuffer: CVPixelBuffer, bbox: CGRect) -> [Float]? {
        guard let model = mobileNetModel else { return nil }

        // Crop and resize to 224×224
        guard let croppedBuffer = cropAndResize(pixelBuffer: pixelBuffer, bbox: bbox, size: 224) else {
            return [Float](repeating: 0, count: GVHMRConstants.imgseqDim)
        }

        // Create MLMultiArray input: [1, 3, 224, 224]
        guard let input = pixelBufferToMLArray(croppedBuffer) else {
            return [Float](repeating: 0, count: GVHMRConstants.imgseqDim)
        }

        do {
            let inputProvider = try MLDictionaryFeatureProvider(dictionary: ["image": input])
            let output = try model.prediction(from: inputProvider)
            // Try all output names — the auto-generated name varies
            for name in output.featureNames.sorted() {
                if let arr = output.featureValue(for: name)?.multiArrayValue,
                   arr.count >= GVHMRConstants.imgseqDim {
                    return multiArrayToFloats(arr, count: GVHMRConstants.imgseqDim)
                }
            }
            // If outputs are smaller, just take the first one
            for name in output.featureNames {
                if let arr = output.featureValue(for: name)?.multiArrayValue {
                    print("[MobileNet] output '\(name)' has \(arr.count) elements (expected \(GVHMRConstants.imgseqDim))")
                    return multiArrayToFloats(arr, count: min(arr.count, GVHMRConstants.imgseqDim))
                }
            }
        } catch {
            print("[MobileNet] inference error: \(error)")
        }

        return [Float](repeating: 0, count: GVHMRConstants.imgseqDim)
    }

    // MARK: - GVHMR Transformer Inference

    /// Run GVHMR student model on a temporal window of observations.
    /// - Parameters:
    ///   - obs: packed keypoints [1, W, 17, 3]
    ///   - cliffCam: packed CLIFF camera [1, W, 3]
    ///   - camAngvel: packed angular velocity [1, W, 6]
    ///   - imgseq: packed image features [1, W, 1024]
    /// - Returns: (pred_x frames [W×151], pred_cam frames [W×3])
    func runGVHMR(obs: [Float], cliffCam: [Float], camAngvel: [Float], imgseq: [Float])
        -> (predX: [[Float]], predCam: [SIMD3<Float>])? {

        guard let model = activeGVHMRModel else {
            print("[GVHMR] Model not loaded (selected: \(activeModelChoice.rawValue))")
            return nil
        }

        let W = GVHMRConstants.windowSize

        do {
            // Create input arrays
            let obsArray = try floatsToMLArray(obs, shape: [1, W, 17, 3])
            let cliffArray = try floatsToMLArray(cliffCam, shape: [1, W, 3])
            let angvelArray = try floatsToMLArray(camAngvel, shape: [1, W, 6])
            let imgseqArray = try floatsToMLArray(imgseq, shape: [1, W, GVHMRConstants.imgseqDim])

            let inputProvider = try MLDictionaryFeatureProvider(dictionary: [
                "obs": obsArray,
                "f_cliffcam": cliffArray,
                "f_cam_angvel": angvelArray,
                "f_imgseq": imgseqArray,
            ])

            let output = try model.prediction(from: inputProvider)

            // Parse pred_x: [1, W, 151]
            // Parse pred_cam: [1, W, 3]
            var predXFrames = [[Float]]()
            var predCamFrames = [SIMD3<Float>]()

            guard let outputs = findGVHMROutputs(output, W: W) else {
                print("[GVHMR] Could not find pred_x/pred_cam outputs. Available: \(output.featureNames.sorted())")
                return nil
            }

            let px = outputs.predX
            let pc = outputs.predCam

            for t in 0..<W {
                var frame = [Float]()
                for c in 0..<GVHMRConstants.outputDim {
                    frame.append(px[[0, t, c] as [NSNumber]].floatValue)
                }
                predXFrames.append(frame)

                predCamFrames.append(SIMD3<Float>(
                    pc[[0, t, 0] as [NSNumber]].floatValue,
                    pc[[0, t, 1] as [NSNumber]].floatValue,
                    pc[[0, t, 2] as [NSNumber]].floatValue
                ))
            }

            return (predXFrames, predCamFrames)

        } catch {
            print("GVHMR inference error: \(error)")
            return nil
        }
    }

    // MARK: - SMPL Forward (Mesh Vertices)

    /// Run the SMPL body model to get 6890 mesh vertices.
    /// - Parameters:
    ///   - bodyPoseAA: 21 joint axis-angle rotations (63 floats)
    ///   - globalOrientAA: root orientation axis-angle (3 floats)
    ///   - betas: shape parameters (10 floats)
    /// - Returns: array of 6890 SIMD3<Float> vertex positions, or nil
    func runSMPL(bodyPoseAA: [Float], globalOrientAA: [Float], betas: [Float]) -> [SIMD3<Float>]? {
        guard let model = smplModel else { return nil }

        do {
            let bpArray = try floatsToMLArray(bodyPoseAA, shape: [1, 63])
            let goArray = try floatsToMLArray(globalOrientAA, shape: [1, 3])
            let betaArray = try floatsToMLArray(betas, shape: [1, 10])

            let inputProvider = try MLDictionaryFeatureProvider(dictionary: [
                "body_pose": bpArray,
                "global_orient": goArray,
                "betas": betaArray,
            ])

            let output = try model.prediction(from: inputProvider)

            // Find the vertex output: shape [1, 6890, 3]
            for name in output.featureNames.sorted() {
                if let arr = output.featureValue(for: name)?.multiArrayValue,
                   arr.count >= 6890 * 3 {
                    let count = 6890
                    var vertices = [SIMD3<Float>]()
                    vertices.reserveCapacity(count)
                    for i in 0..<count {
                        let x = arr[[0, i, 0] as [NSNumber]].floatValue
                        let y = arr[[0, i, 1] as [NSNumber]].floatValue
                        let z = arr[[0, i, 2] as [NSNumber]].floatValue
                        // SMPL outputs in OpenCV camera convention (Y-down, Z-forward).
                        // SceneKit uses Y-up, Z-toward-viewer. Negate Y and Z to convert.
                        vertices.append(SIMD3<Float>(x, -y, -z))
                    }
                    return vertices
                }
            }
        } catch {
            print("[SMPL] inference error: \(error)")
        }
        return nil
    }

    // MARK: - SMPL Forward (Incam Vertices)

    /// Run SMPL to get 6890 mesh vertices in original camera coordinates (Y-down, Z-forward).
    /// Use this for perspective projection overlay on video frames.
    func runSMPLIncam(bodyPoseAA: [Float], globalOrientAA: [Float], betas: [Float]) -> [SIMD3<Float>]? {
        guard let model = smplModel else { return nil }

        do {
            let bpArray = try floatsToMLArray(bodyPoseAA, shape: [1, 63])
            let goArray = try floatsToMLArray(globalOrientAA, shape: [1, 3])
            let betaArray = try floatsToMLArray(betas, shape: [1, 10])

            let inputProvider = try MLDictionaryFeatureProvider(dictionary: [
                "body_pose": bpArray,
                "global_orient": goArray,
                "betas": betaArray,
            ])

            let output = try model.prediction(from: inputProvider)

            for name in output.featureNames.sorted() {
                if let arr = output.featureValue(for: name)?.multiArrayValue,
                   arr.count >= 6890 * 3 {
                    let count = 6890
                    var vertices = [SIMD3<Float>]()
                    vertices.reserveCapacity(count)
                    for i in 0..<count {
                        let x = arr[[0, i, 0] as [NSNumber]].floatValue
                        let y = arr[[0, i, 1] as [NSNumber]].floatValue
                        let z = arr[[0, i, 2] as [NSNumber]].floatValue
                        // Keep original camera coordinates (Y-down, Z-forward)
                        vertices.append(SIMD3<Float>(x, y, z))
                    }
                    return vertices
                }
            }
        } catch {
            print("[SMPL] inference error: \(error)")
        }
        return nil
    }

    // MARK: - Helpers

    private func findOutput(_ output: MLFeatureProvider, names: [String]) -> MLMultiArray? {
        // Try explicit names first
        for name in names {
            if let arr = output.featureValue(for: name)?.multiArrayValue {
                return arr
            }
        }
        return nil
    }

    /// Find the two GVHMR outputs by shape: pred_x is [1,W,151] and pred_cam is [1,W,3]
    private func findGVHMROutputs(_ output: MLFeatureProvider, W: Int)
        -> (predX: MLMultiArray, predCam: MLMultiArray)? {

        // First try named outputs
        let predX = findOutput(output, names: ["pred_x", "var_0", "output_0"])
        let predCam = findOutput(output, names: ["pred_cam", "var_1", "output_1"])
        if let px = predX, let pc = predCam {
            return (px, pc)
        }

        // Fall back: match by shape
        var large: MLMultiArray?  // [1, W, 151]
        var small: MLMultiArray?  // [1, W, 3]
        for name in output.featureNames.sorted() {
            guard let arr = output.featureValue(for: name)?.multiArrayValue else { continue }
            let shape = arr.shape.map { $0.intValue }
            print("[GVHMR] output '\(name)': shape=\(shape)")
            if shape.last == GVHMRConstants.outputDim {
                large = arr
            } else if shape.last == 3 {
                small = arr
            }
        }
        if let px = large, let pc = small {
            return (px, pc)
        }
        return nil
    }

    private func floatsToMLArray(_ floats: [Float], shape: [Int]) throws -> MLMultiArray {
        let nsShape = shape.map { NSNumber(value: $0) }
        let array = try MLMultiArray(shape: nsShape, dataType: .float32)
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: floats.count)
        for i in 0..<floats.count {
            ptr[i] = floats[i]
        }
        return array
    }

    private func multiArrayToFloats(_ array: MLMultiArray, count: Int) -> [Float] {
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: ptr, count: min(count, array.count)))
    }

    private func cropAndResize(pixelBuffer: CVPixelBuffer, bbox: CGRect, size: Int) -> CVPixelBuffer? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let width = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
        let height = CGFloat(CVPixelBufferGetHeight(pixelBuffer))

        // Clamp bbox to image bounds
        let clampedBbox = bbox.intersection(CGRect(x: 0, y: 0, width: width, height: height))
        guard !clampedBbox.isEmpty else { return nil }

        // Square crop
        let side = max(clampedBbox.width, clampedBbox.height)
        let squareBbox = CGRect(
            x: clampedBbox.midX - side / 2,
            y: clampedBbox.midY - side / 2,
            width: side,
            height: side
        ).intersection(CGRect(x: 0, y: 0, width: width, height: height))

        let cropped = ciImage.cropped(to: squareBbox)
        let scaleX = CGFloat(size) / squareBbox.width
        let scaleY = CGFloat(size) / squareBbox.height
        let scaled = cropped
            .transformed(by: CGAffineTransform(translationX: -squareBbox.origin.x,
                                                y: -squareBbox.origin.y))
            .transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))

        let context = CIContext()
        var outputBuffer: CVPixelBuffer?
        CVPixelBufferCreate(nil, size, size, kCVPixelFormatType_32BGRA, nil, &outputBuffer)
        guard let output = outputBuffer else { return nil }
        context.render(scaled, to: output)
        return output
    }

    private func pixelBufferToMLArray(_ pixelBuffer: CVPixelBuffer) -> MLMultiArray? {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)

        guard let array = try? MLMultiArray(shape: [1, 3, NSNumber(value: height), NSNumber(value: width)],
                                            dataType: .float32) else { return nil }

        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: 3 * height * width)
        let pixels = baseAddress.assumingMemoryBound(to: UInt8.self)

        // BGRA → RGB, normalize to [0, 1]
        // ImageNet normalization: (x - mean) / std
        let means: [Float] = [0.485, 0.456, 0.406]
        let stds: [Float] = [0.229, 0.224, 0.225]

        for y in 0..<height {
            for x in 0..<width {
                let offset = y * bytesPerRow + x * 4
                let b = Float(pixels[offset]) / 255.0
                let g = Float(pixels[offset + 1]) / 255.0
                let r = Float(pixels[offset + 2]) / 255.0

                let idx = y * width + x
                ptr[0 * height * width + idx] = (r - means[0]) / stds[0]
                ptr[1 * height * width + idx] = (g - means[1]) / stds[1]
                ptr[2 * height * width + idx] = (b - means[2]) / stds[2]
            }
        }

        return array
    }
}
