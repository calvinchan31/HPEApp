import CoreML
import CoreImage
import Foundation

/// Thin wrapper around an optional CoreML caption model.
///
/// Expected behavior:
/// - Input: image (CVPixelBuffer)
/// - Output: any String feature (preferred), or feature named "caption"
///
/// Model names searched in bundle:
/// - GVHMRCaption
/// - Captioner
/// - MobileCaption
final class CoreMLCaptioner {

    private var model: MLModel?
    private(set) var diagnostics: String = "Caption not loaded"
    private let ciContext = CIContext()

    var isReady: Bool { model != nil }

    func loadModel() {
        let candidates = ["GVHMRCaption", "Captioner", "MobileCaption"]
        var notes = [String]()

        for name in candidates {
            if let loaded = loadModel(named: name, diagnostics: &notes) {
                model = loaded
                diagnostics = notes.joined(separator: "\n")
                return
            }
        }

        model = nil
        notes.append("[MISS] No Caption model found in bundle")
        diagnostics = notes.joined(separator: "\n")
    }

    func caption(pixelBuffer: CVPixelBuffer) -> String? {
        return runCaption(pixelBuffer: pixelBuffer)
    }

    func caption(pixelBuffer: CVPixelBuffer, bbox: CGRect, imageSize: CGSize) -> String? {
        guard let crop = cropAndResize(
            pixelBuffer: pixelBuffer,
            bbox: bbox,
            imageSize: imageSize,
            targetSize: 384
        ) else {
            return runCaption(pixelBuffer: pixelBuffer)
        }
        return runCaption(pixelBuffer: crop)
    }

    private func runCaption(pixelBuffer: CVPixelBuffer) -> String? {
        guard let model else { return nil }

        do {
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "image": MLFeatureValue(pixelBuffer: pixelBuffer)
            ])
            let output = try model.prediction(from: input)

            if let preferred = output.featureValue(for: "caption")?.stringValue,
               !preferred.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return preferred.trimmingCharacters(in: .whitespacesAndNewlines)
            }

            for name in output.featureNames.sorted() {
                if let text = output.featureValue(for: name)?.stringValue {
                    let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !trimmed.isEmpty {
                        return trimmed
                    }
                }
            }
        } catch {
            return nil
        }

        return nil
    }

    private func cropAndResize(
        pixelBuffer: CVPixelBuffer,
        bbox: CGRect,
        imageSize: CGSize,
        targetSize: Int
    ) -> CVPixelBuffer? {
        let frameW = imageSize.width
        let frameH = imageSize.height
        guard frameW > 1, frameH > 1 else { return nil }

        let bounds = CGRect(x: 0, y: 0, width: frameW, height: frameH)
        let clamped = bbox.intersection(bounds)
        guard !clamped.isEmpty else { return nil }

        let side = max(clamped.width, clamped.height)
        let square = CGRect(
            x: clamped.midX - side / 2,
            y: clamped.midY - side / 2,
            width: side,
            height: side
        ).intersection(bounds)
        guard !square.isEmpty else { return nil }

        let ciImage = CIImage(cvPixelBuffer: pixelBuffer).cropped(to: square)
        let scaled = ciImage.transformed(
            by: CGAffineTransform(scaleX: CGFloat(targetSize) / square.width,
                                  y: CGFloat(targetSize) / square.height)
        )

        var out: CVPixelBuffer?
        CVPixelBufferCreate(
            nil,
            targetSize,
            targetSize,
            kCVPixelFormatType_32BGRA,
            nil,
            &out
        )
        guard let out else { return nil }

        ciContext.render(scaled, to: out)
        return out
    }

    private func loadModel(named name: String, diagnostics: inout [String]) -> MLModel? {
        let exts = ["mlmodelc", "mlpackage"]
        for ext in exts {
            guard let url = Bundle.main.url(forResource: name, withExtension: ext) else {
                continue
            }
            do {
                let config = MLModelConfiguration()
                config.computeUnits = .cpuAndNeuralEngine
                let loaded = try MLModel(contentsOf: url, configuration: config)
                diagnostics.append("[OK] \(name).\(ext)")
                diagnostics.append("  outputs: \(loaded.modelDescription.outputDescriptionsByName.keys.sorted())")
                return loaded
            } catch {
                diagnostics.append("[ERR] \(name).\(ext): \(error.localizedDescription)")
            }
        }
        diagnostics.append("[MISS] \(name) not found")
        return nil
    }
}