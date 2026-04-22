import AVFoundation
import CoreImage
import UIKit

/// Renders GVHMR results into output videos matching the demo format:
///   1_incam.mp4  — SMPL mesh overlay on original video
///   2_global.mp4 — SMPL in world coordinates (front view)
/// Also exports hmr4d_results.json with per-frame SMPL parameters.
class VideoExporter {

    /// Export incam video: original frames with SMPL mesh overlay using perspective projection.
    static func exportIncamVideo(
        sourceURL: URL,
        results: [VideoProcessor.FrameResult],
        smplFaces: [UInt32],
        focalLength: Float,
        outputURL: URL,
        completion: @escaping (Bool, String) -> Void
    ) {
        guard !results.isEmpty else {
            completion(false, "No results to export")
            return
        }

        let asset = AVURLAsset(url: sourceURL)
        guard let track = asset.tracks(withMediaType: .video).first else {
            completion(false, "No video track")
            return
        }

        let size = track.naturalSize.applying(track.preferredTransform)
        let videoW = Int(abs(size.width))
        let videoH = Int(abs(size.height))
        let frameRate = track.nominalFrameRate

        // Setup writer
        guard let writer = try? AVAssetWriter(outputURL: outputURL, fileType: .mp4) else {
            completion(false, "Cannot create writer")
            return
        }

        let outputSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: videoW,
            AVVideoHeightKey: videoH,
        ]
        let writerInput = AVAssetWriterInput(mediaType: .video, outputSettings: outputSettings)
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: writerInput,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: videoW,
                kCVPixelBufferHeightKey as String: videoH,
            ]
        )
        writer.add(writerInput)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)

        // Read source frames
        guard let reader = try? AVAssetReader(asset: asset) else {
            completion(false, "Cannot create reader")
            return
        }
        let readOutput = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ])
        readOutput.alwaysCopiesSampleData = false
        reader.add(readOutput)
        reader.startReading()

        let icx = Float(videoW) / 2
        let icy = Float(videoH) / 2
        var frameIdx = 0

        while let sampleBuffer = readOutput.copyNextSampleBuffer(),
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {

            guard frameIdx < results.count else { break }

            // Create a writable copy for drawing
            CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
            let w = CVPixelBufferGetWidth(pixelBuffer)
            let h = CVPixelBufferGetHeight(pixelBuffer)

            var outputBuffer: CVPixelBuffer?
            CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, nil, &outputBuffer)
            guard let outBuf = outputBuffer else {
                frameIdx += 1
                CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
                continue
            }

            // Copy source frame
            CVPixelBufferLockBaseAddress(outBuf, [])
            let srcPtr = CVPixelBufferGetBaseAddress(pixelBuffer)!
            let dstPtr = CVPixelBufferGetBaseAddress(outBuf)!
            let srcBPR = CVPixelBufferGetBytesPerRow(pixelBuffer)
            let dstBPR = CVPixelBufferGetBytesPerRow(outBuf)
            for row in 0..<h {
                memcpy(dstPtr.advanced(by: row * dstBPR), srcPtr.advanced(by: row * srcBPR), min(srcBPR, dstBPR))
            }
            CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)

            // Draw SMPL mesh overlay with perspective projection
            let result = results[frameIdx]
            if let incamVerts = result.gvhmrResult.meshVerticesIncam,
               let transl = result.gvhmrResult.translFullCam,
               !smplFaces.isEmpty {
                var zBuffer = [Float](repeating: Float.greatestFiniteMagnitude, count: w * h)
                drawMeshOverlay(
                    buffer: outBuf,
                    vertices: incamVerts,
                    translation: transl,
                    faces: smplFaces,
                    focalLength: focalLength,
                    icx: icx, icy: icy,
                    width: w, height: h,
                    zBuffer: &zBuffer
                )
            }

            CVPixelBufferUnlockBaseAddress(outBuf, [])

            // Write frame
            let pts = CMTime(value: CMTimeValue(frameIdx), timescale: CMTimeScale(frameRate))
            while !writerInput.isReadyForMoreMediaData {
                Thread.sleep(forTimeInterval: 0.01)
            }
            adaptor.append(outBuf, withPresentationTime: pts)

            frameIdx += 1
        }

        writerInput.markAsFinished()
        writer.finishWriting {
            let success = writer.status == .completed
            completion(success, success ? "OK" : (writer.error?.localizedDescription ?? "Unknown error"))
        }
    }

    /// Export a global view video using mesh rendering from a fixed virtual camera.
    /// Falls back to skeleton rendering when mesh vertices are unavailable.
    static func exportGlobalVideo(
        results: [VideoProcessor.FrameResult],
        smplFaces: [UInt32],
        videoSize: CGSize,
        fps: Double,
        outputURL: URL,
        completion: @escaping (Bool, String) -> Void
    ) {
        guard !results.isEmpty else {
            completion(false, "No results")
            return
        }

        let side = 512  // square output
        guard let writer = try? AVAssetWriter(outputURL: outputURL, fileType: .mp4) else {
            completion(false, "Cannot create writer")
            return
        }

        let outputSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: side,
            AVVideoHeightKey: side,
        ]
        let writerInput = AVAssetWriterInput(mediaType: .video, outputSettings: outputSettings)
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: writerInput,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: side,
                kCVPixelBufferHeightKey as String: side,
            ]
        )
        writer.add(writerInput)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)

        for (i, frameResult) in results.enumerated() {
            var buffer: CVPixelBuffer?
            CVPixelBufferCreate(nil, side, side, kCVPixelFormatType_32BGRA, nil, &buffer)
            guard let buf = buffer else { continue }

            CVPixelBufferLockBaseAddress(buf, [])

            // Clear to dark gray
            let ptr = CVPixelBufferGetBaseAddress(buf)!.assumingMemoryBound(to: UInt8.self)
            let bpr = CVPixelBufferGetBytesPerRow(buf)
            for row in 0..<side {
                for col in 0..<side {
                    let off = row * bpr + col * 4
                    ptr[off] = 40      // B
                    ptr[off + 1] = 40  // G
                    ptr[off + 2] = 40  // R
                    ptr[off + 3] = 255 // A
                }
            }

            if let verts = frameResult.gvhmrResult.meshVertices,
               !verts.isEmpty,
               !smplFaces.isEmpty {
                var zBuffer = [Float](repeating: Float.greatestFiniteMagnitude, count: side * side)
                drawGlobalMesh(
                    buffer: buf,
                    vertices: verts,
                    translation: nil,
                    faces: smplFaces,
                    width: side,
                    height: side,
                    zBuffer: &zBuffer
                )
            } else {
                // Fallback for incomplete outputs that only contain joints.
                let joints = frameResult.gvhmrResult.joints3D
                var projected = [CGPoint]()
                let scale: Float = 250.0
                let cx = Float(side) / 2
                let cy = Float(side) / 2 - 50

                for j in joints {
                    let px = cx + j.x * scale
                    let py = cy + j.y * scale
                    projected.append(CGPoint(x: CGFloat(px), y: CGFloat(py)))
                }

                drawSkeleton(buffer: buf, joints2D: projected, width: side, height: side)
            }

            CVPixelBufferUnlockBaseAddress(buf, [])

            let pts = CMTime(value: CMTimeValue(i), timescale: CMTimeScale(fps))
            while !writerInput.isReadyForMoreMediaData {
                Thread.sleep(forTimeInterval: 0.01)
            }
            adaptor.append(buf, withPresentationTime: pts)
        }

        writerInput.markAsFinished()
        writer.finishWriting {
            let success = writer.status == .completed
            completion(success, success ? "OK" : (writer.error?.localizedDescription ?? "Unknown error"))
        }
    }

    /// Export a global view video with multiple persons rendered in distinct colors.
    static func exportGlobalVideoMulti(
        multiResults: [MultiPersonFrameResult],
        smplFaces: [UInt32],
        videoSize: CGSize,
        fps: Double,
        outputURL: URL,
        completion: @escaping (Bool, String) -> Void
    ) {
        guard !multiResults.isEmpty else {
            completion(false, "No results")
            return
        }

        let side = 512
        guard let writer = try? AVAssetWriter(outputURL: outputURL, fileType: .mp4) else {
            completion(false, "Cannot create writer")
            return
        }

        let outputSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: side,
            AVVideoHeightKey: side,
        ]
        let writerInput = AVAssetWriterInput(mediaType: .video, outputSettings: outputSettings)
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: writerInput,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: side,
                kCVPixelBufferHeightKey as String: side,
            ]
        )
        writer.add(writerInput)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)

        for (i, multiFrame) in multiResults.enumerated() {
            var buffer: CVPixelBuffer?
            CVPixelBufferCreate(nil, side, side, kCVPixelFormatType_32BGRA, nil, &buffer)
            guard let buf = buffer else { continue }

            CVPixelBufferLockBaseAddress(buf, [])

            let ptr = CVPixelBufferGetBaseAddress(buf)!.assumingMemoryBound(to: UInt8.self)
            let bpr = CVPixelBufferGetBytesPerRow(buf)
            for row in 0..<side {
                for col in 0..<side {
                    let off = row * bpr + col * 4
                    ptr[off] = 40
                    ptr[off + 1] = 40
                    ptr[off + 2] = 40
                    ptr[off + 3] = 255
                }
            }

            var zBuffer = [Float](repeating: Float.greatestFiniteMagnitude, count: side * side)
            for person in multiFrame.persons {
                guard let verts = person.gvhmrResult.meshVertices,
                      !verts.isEmpty,
                      !smplFaces.isEmpty else { continue }

                let rgb = PersonColors.color(for: person.trackID)
                let meshColor = (UInt8(rgb.0 * 255), UInt8(rgb.1 * 255), UInt8(rgb.2 * 255))
                drawGlobalMesh(
                    buffer: buf,
                    vertices: verts,
                    translation: person.gvhmrResult.translFullCam,
                    faces: smplFaces,
                    width: side,
                    height: side,
                    zBuffer: &zBuffer,
                    meshColor: meshColor
                )
            }

            CVPixelBufferUnlockBaseAddress(buf, [])

            let pts = CMTime(value: CMTimeValue(i), timescale: CMTimeScale(fps))
            while !writerInput.isReadyForMoreMediaData {
                Thread.sleep(forTimeInterval: 0.01)
            }
            adaptor.append(buf, withPresentationTime: pts)
        }

        writerInput.markAsFinished()
        writer.finishWriting {
            let success = writer.status == .completed
            completion(success, success ? "OK" : (writer.error?.localizedDescription ?? "Unknown error"))
        }
    }

    /// Export a labeled compare video with Original input + Small/Medium/Teacher outputs.
    static func exportComparisonLabeledVideo(
        sourceURL: URL,
        comparisonFrameResults: [GVHMRModelChoice: [VideoProcessor.FrameResult]],
        comparisonMultiPersonResults: [GVHMRModelChoice: [MultiPersonFrameResult]],
        fps: Double,
        outputURL: URL,
        completion: @escaping (Bool, String) -> Void
    ) {
        let orderedModels: [GVHMRModelChoice] = [.small, .medium, .original]
        let maxFrames = orderedModels
            .map { max(comparisonFrameResults[$0]?.count ?? 0, comparisonMultiPersonResults[$0]?.count ?? 0) }
            .max() ?? 0

        guard maxFrames > 0 else {
            completion(false, "No comparison frames to export")
            return
        }

        let asset = AVURLAsset(url: sourceURL)
        guard let track = asset.tracks(withMediaType: .video).first else {
            completion(false, "No video track")
            return
        }

        guard let reader = try? AVAssetReader(asset: asset) else {
            completion(false, "Cannot create reader")
            return
        }

        let readOutput = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        ])
        readOutput.alwaysCopiesSampleData = false
        reader.add(readOutput)
        reader.startReading()

        let width = 1280
        let height = 720
        guard let writer = try? AVAssetWriter(outputURL: outputURL, fileType: .mp4) else {
            completion(false, "Cannot create writer")
            return
        }

        let outputSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
        ]
        let writerInput = AVAssetWriterInput(mediaType: .video, outputSettings: outputSettings)
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: writerInput,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
            ]
        )

        writer.add(writerInput)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)

        let margin: CGFloat = 16
        let panelW = (CGFloat(width) - margin * 3) / 2
        let panelH = (CGFloat(height) - margin * 3) / 2
        let topY = margin
        let bottomY = margin * 2 + panelH

        let originalRect = CGRect(x: margin, y: topY, width: panelW, height: panelH)
        let smallRect = CGRect(x: margin * 2 + panelW, y: topY, width: panelW, height: panelH)
        let mediumRect = CGRect(x: margin, y: bottomY, width: panelW, height: panelH)
        let teacherRect = CGRect(x: margin * 2 + panelW, y: bottomY, width: panelW, height: panelH)

        let ciContext = CIContext()
        var frameIndex = 0

        while let sampleBuffer = readOutput.copyNextSampleBuffer(),
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer),
              frameIndex < maxFrames {

            var buffer: CVPixelBuffer?
            CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA, nil, &buffer)
            guard let buf = buffer else {
                frameIndex += 1
                continue
            }

            let srcW = CVPixelBufferGetWidth(pixelBuffer)
            let srcH = CVPixelBufferGetHeight(pixelBuffer)
            let srcRect = CGRect(x: 0, y: 0, width: srcW, height: srcH)
            let cgImage = ciContext.createCGImage(CIImage(cvPixelBuffer: pixelBuffer), from: srcRect)

            CVPixelBufferLockBaseAddress(buf, [])
            guard let ptr = CVPixelBufferGetBaseAddress(buf) else {
                CVPixelBufferUnlockBaseAddress(buf, [])
                frameIndex += 1
                continue
            }
            let bpr = CVPixelBufferGetBytesPerRow(buf)

            guard let cg = CGContext(
                data: ptr,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: bpr,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
            ) else {
                CVPixelBufferUnlockBaseAddress(buf, [])
                frameIndex += 1
                continue
            }

            cg.translateBy(x: 0, y: CGFloat(height))
            cg.scaleBy(x: 1, y: -1)

            cg.setFillColor(UIColor(white: 0.08, alpha: 1).cgColor)
            cg.fill(CGRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(height)))

            drawComparePanelBackground(cg: cg, rect: originalRect, title: "Original Video")
            if let cgImage {
                drawVideoFrameUpright(cg: cg, image: cgImage, rect: originalRect.insetBy(dx: 6, dy: 28))
            }

            drawLabeledIncamPanel(
                cg: cg,
                rect: smallRect,
                title: "Small Output",
                model: .small,
                frameIndex: frameIndex,
                comparisonFrameResults: comparisonFrameResults,
                comparisonMultiPersonResults: comparisonMultiPersonResults,
                sourceImage: cgImage,
                sourceSize: CGSize(width: srcW, height: srcH),
                pointColor: UIColor(red: 0.30, green: 0.88, blue: 0.95, alpha: 1)
            )

            drawLabeledIncamPanel(
                cg: cg,
                rect: mediumRect,
                title: "Medium Output",
                model: .medium,
                frameIndex: frameIndex,
                comparisonFrameResults: comparisonFrameResults,
                comparisonMultiPersonResults: comparisonMultiPersonResults,
                sourceImage: cgImage,
                sourceSize: CGSize(width: srcW, height: srcH),
                pointColor: UIColor(red: 0.46, green: 0.90, blue: 0.46, alpha: 1)
            )

            drawLabeledIncamPanel(
                cg: cg,
                rect: teacherRect,
                title: "Teacher Output",
                model: .original,
                frameIndex: frameIndex,
                comparisonFrameResults: comparisonFrameResults,
                comparisonMultiPersonResults: comparisonMultiPersonResults,
                sourceImage: cgImage,
                sourceSize: CGSize(width: srcW, height: srcH),
                pointColor: UIColor(red: 1.0, green: 0.72, blue: 0.33, alpha: 1)
            )

            let footerAttr: [NSAttributedString.Key: Any] = [
                .font: UIFont.monospacedSystemFont(ofSize: 11, weight: .regular),
                .foregroundColor: UIColor.white.withAlphaComponent(0.75),
            ]
            NSString(string: "Frame \(min(frameIndex + 1, maxFrames))/\(maxFrames)").draw(
                in: CGRect(x: 14, y: CGFloat(height) - 24, width: 220, height: 16),
                withAttributes: footerAttr
            )

            CVPixelBufferUnlockBaseAddress(buf, [])

            let pts = CMTime(value: CMTimeValue(frameIndex), timescale: CMTimeScale(max(fps, 1)))
            while !writerInput.isReadyForMoreMediaData {
                Thread.sleep(forTimeInterval: 0.005)
            }
            adaptor.append(buf, withPresentationTime: pts)
            frameIndex += 1
        }

        writerInput.markAsFinished()
        writer.finishWriting {
            let success = writer.status == .completed
            completion(success, success ? "OK" : (writer.error?.localizedDescription ?? "Unknown error"))
        }
    }

    /// Export one compare SMPL video with original video + Small/Medium/Teacher outputs in a 2x2 layout.
    static func exportComparisonCompositeVideo(
        sourceURL: URL,
        comparisonFrameResults: [GVHMRModelChoice: [VideoProcessor.FrameResult]],
        comparisonMultiPersonResults: [GVHMRModelChoice: [MultiPersonFrameResult]],
        benchmarks: [PerformanceMetrics.ModelBenchmark],
        smplFaces: [UInt32],
        fps: Double,
        outputURL: URL,
        completion: @escaping (Bool, String) -> Void
    ) {
        let orderedModels: [GVHMRModelChoice] = [.small, .medium, .original]
        let maxFrames = orderedModels
            .map { max(comparisonFrameResults[$0]?.count ?? 0, comparisonMultiPersonResults[$0]?.count ?? 0) }
            .max() ?? 0

        guard maxFrames > 0 else {
            completion(false, "No comparison frames to export")
            return
        }

        let asset = AVURLAsset(url: sourceURL)
        guard let track = asset.tracks(withMediaType: .video).first else {
            completion(false, "No video track")
            return
        }

        guard let reader = try? AVAssetReader(asset: asset) else {
            completion(false, "Cannot create reader")
            return
        }

        let readOutput = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
        ])
        readOutput.alwaysCopiesSampleData = false
        reader.add(readOutput)
        reader.startReading()

        let width = 1280
        let height = 720

        guard let writer = try? AVAssetWriter(outputURL: outputURL, fileType: .mp4) else {
            completion(false, "Cannot create writer")
            return
        }

        let outputSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
        ]
        let writerInput = AVAssetWriterInput(mediaType: .video, outputSettings: outputSettings)
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: writerInput,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
            ]
        )

        writer.add(writerInput)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)

        let margin: CGFloat = 16
        let panelW = (CGFloat(width) - margin * 3) / 2
        let panelH = (CGFloat(height) - margin * 3) / 2
        let topY = margin
        let bottomY = margin * 2 + panelH

        let originalRect = CGRect(x: margin, y: topY, width: panelW, height: panelH)
        let panelRects: [GVHMRModelChoice: CGRect] = [
            .small: CGRect(x: margin * 2 + panelW, y: topY, width: panelW, height: panelH),
            .medium: CGRect(x: margin, y: bottomY, width: panelW, height: panelH),
            .original: CGRect(x: margin * 2 + panelW, y: bottomY, width: panelW, height: panelH),
        ]

        let ciContext = CIContext()
        var frameIndex = 0

        while let sampleBuffer = readOutput.copyNextSampleBuffer(),
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer),
              frameIndex < maxFrames {
            var buffer: CVPixelBuffer?
            CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA, nil, &buffer)
            guard let buf = buffer else { continue }

            let srcW = CVPixelBufferGetWidth(pixelBuffer)
            let srcH = CVPixelBufferGetHeight(pixelBuffer)
            let srcRect = CGRect(x: 0, y: 0, width: srcW, height: srcH)
            let cgImage = ciContext.createCGImage(CIImage(cvPixelBuffer: pixelBuffer), from: srcRect)

            CVPixelBufferLockBaseAddress(buf, [])
            guard let ptr = CVPixelBufferGetBaseAddress(buf) else {
                CVPixelBufferUnlockBaseAddress(buf, [])
                continue
            }
            let bpr = CVPixelBufferGetBytesPerRow(buf)

            guard let cg = CGContext(
                data: ptr,
                width: width,
                height: height,
                bitsPerComponent: 8,
                bytesPerRow: bpr,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
            ) else {
                CVPixelBufferUnlockBaseAddress(buf, [])
                continue
            }

            // UIKit-style top-left origin
            cg.translateBy(x: 0, y: CGFloat(height))
            cg.scaleBy(x: 1, y: -1)

            cg.setFillColor(UIColor(white: 0.08, alpha: 1).cgColor)
            cg.fill(CGRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(height)))

            drawComparePanelBackground(cg: cg, rect: originalRect, title: "Original Video")
            if let cgImage {
                drawVideoFrameUpright(cg: cg, image: cgImage, rect: originalRect.insetBy(dx: 6, dy: 28))
            }

            for model in orderedModels {
                guard let rect = panelRects[model] else { continue }
                let panelTitle: String
                switch model {
                case .small:
                    panelTitle = "Small"
                case .medium:
                    panelTitle = "Medium"
                case .original:
                    panelTitle = "Teacher"
                }
                drawComparePanelBackground(cg: cg, rect: rect, title: panelTitle)

                if let multiFrames = comparisonMultiPersonResults[model],
                   frameIndex < multiFrames.count,
                   !multiFrames[frameIndex].persons.isEmpty {
                    for person in multiFrames[frameIndex].persons {
                        guard let verts = person.gvhmrResult.meshVertices else { continue }
                        let rgb = PersonColors.color(for: person.trackID)
                        let color = UIColor(red: CGFloat(rgb.0), green: CGFloat(rgb.1), blue: CGFloat(rgb.2), alpha: 1)
                        drawGlobalPointCloud(
                            cg: cg,
                            rect: rect,
                            vertices: verts,
                            translation: person.gvhmrResult.translFullCam,
                            color: color
                        )
                    }
                } else if let frames = comparisonFrameResults[model],
                          frameIndex < frames.count,
                          let verts = frames[frameIndex].gvhmrResult.meshVertices {
                    drawGlobalPointCloud(
                        cg: cg,
                        rect: rect,
                        vertices: verts,
                        translation: frames[frameIndex].gvhmrResult.translFullCam,
                        color: UIColor(red: 0.83, green: 0.88, blue: 0.96, alpha: 1)
                    )
                }
            }

            let footerAttr: [NSAttributedString.Key: Any] = [
                .font: UIFont.monospacedSystemFont(ofSize: 11, weight: .regular),
                .foregroundColor: UIColor.white.withAlphaComponent(0.75),
            ]
            NSString(string: "Frame \(min(frameIndex + 1, maxFrames))/\(maxFrames) | SMPL: \(!smplFaces.isEmpty ? "on" : "off")").draw(
                in: CGRect(x: 14, y: CGFloat(height) - 24, width: 340, height: 16),
                withAttributes: footerAttr
            )

            CVPixelBufferUnlockBaseAddress(buf, [])

            let pts = CMTime(value: CMTimeValue(frameIndex), timescale: CMTimeScale(max(fps, 1)))
            while !writerInput.isReadyForMoreMediaData {
                Thread.sleep(forTimeInterval: 0.005)
            }
            adaptor.append(buf, withPresentationTime: pts)
            frameIndex += 1
        }

        writerInput.markAsFinished()
        writer.finishWriting {
            let success = writer.status == .completed
            completion(success, success ? "OK" : (writer.error?.localizedDescription ?? "Unknown error"))
        }
    }

    private static func drawLabeledIncamPanel(
        cg: CGContext,
        rect: CGRect,
        title: String,
        model: GVHMRModelChoice,
        frameIndex: Int,
        comparisonFrameResults: [GVHMRModelChoice: [VideoProcessor.FrameResult]],
        comparisonMultiPersonResults: [GVHMRModelChoice: [MultiPersonFrameResult]],
        sourceImage: CGImage?,
        sourceSize: CGSize,
        pointColor: UIColor
    ) {
        drawComparePanelBackground(cg: cg, rect: rect, title: title)
        let contentRect = rect.insetBy(dx: 6, dy: 28)

        if let sourceImage {
            drawVideoFrameUpright(cg: cg, image: sourceImage, rect: contentRect)
        } else {
            cg.setFillColor(UIColor(white: 0.12, alpha: 1).cgColor)
            cg.fill(contentRect)
        }

        if let multiFrames = comparisonMultiPersonResults[model],
           frameIndex < multiFrames.count,
           !multiFrames[frameIndex].persons.isEmpty {
            for person in multiFrames[frameIndex].persons {
                guard let verts = person.gvhmrResult.meshVerticesIncam,
                      let transl = person.gvhmrResult.translFullCam else { continue }

                let rgb = PersonColors.color(for: person.trackID)
                let color = UIColor(red: CGFloat(rgb.0), green: CGFloat(rgb.1), blue: CGFloat(rgb.2), alpha: 1)
                drawIncamPointCloud(
                    cg: cg,
                    rect: contentRect,
                    vertices: verts,
                    translation: transl,
                    sourceSize: sourceSize,
                    pointColor: color
                )
            }
            return
        }

        if let frames = comparisonFrameResults[model],
           frameIndex < frames.count,
           let verts = frames[frameIndex].gvhmrResult.meshVerticesIncam,
           let transl = frames[frameIndex].gvhmrResult.translFullCam {
            drawIncamPointCloud(
                cg: cg,
                rect: contentRect,
                vertices: verts,
                translation: transl,
                sourceSize: sourceSize,
                pointColor: pointColor
            )
        }
    }

    private static func drawIncamPointCloud(
        cg: CGContext,
        rect: CGRect,
        vertices: [SIMD3<Float>],
        translation: SIMD3<Float>,
        sourceSize: CGSize,
        pointColor: UIColor
    ) {
        guard !vertices.isEmpty, sourceSize.width > 1, sourceSize.height > 1 else { return }

        let fl = Float(sourceSize.width) / (2.0 * tan(60.0 * .pi / 360.0))
        let icx = Float(sourceSize.width) * 0.5
        let icy = Float(sourceSize.height) * 0.5
        let sx = Float(rect.width) / Float(sourceSize.width)
        let sy = Float(rect.height) / Float(sourceSize.height)

        cg.setFillColor(pointColor.withAlphaComponent(0.8).cgColor)
        for v in vertices {
            let vx = v.x + translation.x
            let vy = v.y + translation.y
            let vz = v.z + translation.z
            if vz <= 0.01 { continue }

            let px = CGFloat(fl * vx / vz + icx)
            let py = CGFloat(fl * vy / vz + icy)

            let x = rect.minX + px * CGFloat(sx)
            let y = rect.minY + py * CGFloat(sy)
            if rect.contains(CGPoint(x: x, y: y)) {
                cg.fillEllipse(in: CGRect(x: x - 1.1, y: y - 1.1, width: 2.2, height: 2.2))
            }
        }
    }

    /// Draw source frame upright inside a top-left-origin drawing context.
    private static func drawVideoFrameUpright(cg: CGContext, image: CGImage, rect: CGRect) {
        cg.saveGState()
        cg.translateBy(x: rect.minX, y: rect.minY + rect.height)
        cg.scaleBy(x: 1, y: -1)
        cg.draw(image, in: CGRect(x: 0, y: 0, width: rect.width, height: rect.height))
        cg.restoreGState()
    }

    /// Export per-frame results as JSON (similar to hmr4d_results.pt).
    static func exportResults(
        results: [VideoProcessor.FrameResult],
        videoSize: CGSize,
        fps: Double,
        outputURL: URL,
        captionTimeline: [CaptionSemanticFrame] = []
    ) -> Bool {
        let captionByFrame = Dictionary(uniqueKeysWithValues: captionTimeline.map { ($0.frameIndex, $0) })
        var frames = [[String: Any]]()
        for r in results {
            let gr = r.gvhmrResult
            var frame: [String: Any] = [
                "frame": r.frameIndex,
                "body_pose_aa": gr.bodyPoseAA.flatMap { [$0.x, $0.y, $0.z] },
                "global_orient_aa": [gr.globalOrient.x, gr.globalOrient.y, gr.globalOrient.z],
                "betas": gr.betas,
                "pred_cam": [gr.predCam.x, gr.predCam.y, gr.predCam.z],
                "pred_x": r.predX,
                "joints3D": gr.joints3D.flatMap { [$0.x, $0.y, $0.z] },
            ]
            if let bbox = Optional(r.bbox) {
                frame["bbox"] = [bbox.origin.x, bbox.origin.y, bbox.width, bbox.height]
            }
            frame["bbx_xys"] = [r.bbxXYS.x, r.bbxXYS.y, r.bbxXYS.z]
            if let t = gr.translFullCam {
                frame["transl_full_cam"] = [t.x, t.y, t.z]
            }
            if let semantic = captionByFrame[r.frameIndex] {
                frame["caption"] = [
                    "caption": semantic.caption,
                    "confidence": semantic.confidence,
                    "source": semantic.source,
                    "action_tag": semantic.actionTag,
                    "timestamp_sec": semantic.timestampSec,
                ]
            }
            frames.append(frame)
        }

        var output: [String: Any] = [
            "video_width": videoSize.width,
            "video_height": videoSize.height,
            "fps": fps,
            "num_frames": results.count,
            "frames": frames,
        ]
        if !captionTimeline.isEmpty {
            output["caption"] = [
                "source": captionTimeline.first?.source ?? "unknown",
                "num_captions": captionTimeline.count,
            ]
        }

        guard let data = try? JSONSerialization.data(withJSONObject: output, options: .prettyPrinted) else {
            return false
        }
        return FileManager.default.createFile(atPath: outputURL.path, contents: data)
    }

    // MARK: - Multi-Person Export

    /// Export incam video with multiple colored SMPL meshes per frame.
    static func exportIncamVideoMulti(
        sourceURL: URL,
        multiResults: [MultiPersonFrameResult],
        smplFaces: [UInt32],
        focalLength: Float,
        outputURL: URL,
        completion: @escaping (Bool, String) -> Void
    ) {
        guard !multiResults.isEmpty else {
            completion(false, "No results to export")
            return
        }

        let asset = AVURLAsset(url: sourceURL)
        guard let track = asset.tracks(withMediaType: .video).first else {
            completion(false, "No video track")
            return
        }

        let size = track.naturalSize.applying(track.preferredTransform)
        let videoW = Int(abs(size.width))
        let videoH = Int(abs(size.height))
        let frameRate = track.nominalFrameRate

        guard let writer = try? AVAssetWriter(outputURL: outputURL, fileType: .mp4) else {
            completion(false, "Cannot create writer")
            return
        }

        let outputSettings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: videoW,
            AVVideoHeightKey: videoH,
        ]
        let writerInput = AVAssetWriterInput(mediaType: .video, outputSettings: outputSettings)
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: writerInput,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: videoW,
                kCVPixelBufferHeightKey as String: videoH,
            ]
        )
        writer.add(writerInput)
        writer.startWriting()
        writer.startSession(atSourceTime: .zero)

        guard let reader = try? AVAssetReader(asset: asset) else {
            completion(false, "Cannot create reader")
            return
        }
        let readOutput = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ])
        readOutput.alwaysCopiesSampleData = false
        reader.add(readOutput)
        reader.startReading()

        let icx = Float(videoW) / 2
        let icy = Float(videoH) / 2
        var frameIdx = 0

        while let sampleBuffer = readOutput.copyNextSampleBuffer(),
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {

            guard frameIdx < multiResults.count else { break }

            CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
            let w = CVPixelBufferGetWidth(pixelBuffer)
            let h = CVPixelBufferGetHeight(pixelBuffer)

            var outputBuffer: CVPixelBuffer?
            CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, nil, &outputBuffer)
            guard let outBuf = outputBuffer else {
                frameIdx += 1
                CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
                continue
            }

            CVPixelBufferLockBaseAddress(outBuf, [])
            let srcPtr = CVPixelBufferGetBaseAddress(pixelBuffer)!
            let dstPtr = CVPixelBufferGetBaseAddress(outBuf)!
            let srcBPR = CVPixelBufferGetBytesPerRow(pixelBuffer)
            let dstBPR = CVPixelBufferGetBytesPerRow(outBuf)
            for row in 0..<h {
                memcpy(dstPtr.advanced(by: row * dstBPR), srcPtr.advanced(by: row * srcBPR), min(srcBPR, dstBPR))
            }
            CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)

            // Draw all persons with shared z-buffer for correct occlusion
            let multiFrame = multiResults[frameIdx]
            if !smplFaces.isEmpty {
                var zBuffer = [Float](repeating: Float.greatestFiniteMagnitude, count: w * h)
                for person in multiFrame.persons {
                    if let incamVerts = person.gvhmrResult.meshVerticesIncam,
                       let transl = person.gvhmrResult.translFullCam {
                        let rgb = PersonColors.color(for: person.trackID)
                        let meshColor = (UInt8(rgb.0 * 255), UInt8(rgb.1 * 255), UInt8(rgb.2 * 255))
                        drawMeshOverlay(
                            buffer: outBuf,
                            vertices: incamVerts,
                            translation: transl,
                            faces: smplFaces,
                            focalLength: focalLength,
                            icx: icx, icy: icy,
                            width: w, height: h,
                            zBuffer: &zBuffer,
                            meshColor: meshColor
                        )
                    }
                }
            }

            CVPixelBufferUnlockBaseAddress(outBuf, [])

            let pts = CMTime(value: CMTimeValue(frameIdx), timescale: CMTimeScale(frameRate))
            while !writerInput.isReadyForMoreMediaData {
                Thread.sleep(forTimeInterval: 0.01)
            }
            adaptor.append(outBuf, withPresentationTime: pts)

            frameIdx += 1
        }

        writerInput.markAsFinished()
        writer.finishWriting {
            let success = writer.status == .completed
            completion(success, success ? "OK" : (writer.error?.localizedDescription ?? "Unknown error"))
        }
    }

    /// Export per-person, per-frame results as JSON for multi-person mode.
    static func exportResultsMulti(
        multiResults: [MultiPersonFrameResult],
        videoSize: CGSize,
        fps: Double,
        outputURL: URL,
        captionTimeline: [CaptionSemanticFrame] = []
    ) -> Bool {
        let captionByFrame = Dictionary(uniqueKeysWithValues: captionTimeline.map { ($0.frameIndex, $0) })
        var frames = [[String: Any]]()
        for mf in multiResults {
            var personsArr = [[String: Any]]()
            for p in mf.persons {
                let gr = p.gvhmrResult
                var personDict: [String: Any] = [
                    "person_index": p.personIndex,
                    "track_id": p.trackID,
                    "body_pose_aa": gr.bodyPoseAA.flatMap { [$0.x, $0.y, $0.z] },
                    "global_orient_aa": [gr.globalOrient.x, gr.globalOrient.y, gr.globalOrient.z],
                    "betas": gr.betas,
                    "pred_cam": [gr.predCam.x, gr.predCam.y, gr.predCam.z],
                    "pred_x": p.predX,
                    "joints3D": gr.joints3D.flatMap { [$0.x, $0.y, $0.z] },
                    "bbox": [p.bbox.origin.x, p.bbox.origin.y, p.bbox.width, p.bbox.height],
                    "bbx_xys": [p.bbxXYS.x, p.bbxXYS.y, p.bbxXYS.z],
                ]
                if let t = gr.translFullCam {
                    personDict["transl_full_cam"] = [t.x, t.y, t.z]
                }
                personsArr.append(personDict)
            }
            var framePayload: [String: Any] = [
                "frame": mf.frameIndex,
                "persons": personsArr,
            ]
            if let semantic = captionByFrame[mf.frameIndex] {
                framePayload["caption"] = [
                    "caption": semantic.caption,
                    "confidence": semantic.confidence,
                    "source": semantic.source,
                    "action_tag": semantic.actionTag,
                    "timestamp_sec": semantic.timestampSec,
                ]
            }
            frames.append(framePayload)
        }

        var output: [String: Any] = [
            "video_width": videoSize.width,
            "video_height": videoSize.height,
            "fps": fps,
            "num_frames": multiResults.count,
            "multi_person": true,
            "frames": frames,
        ]
        if !captionTimeline.isEmpty {
            output["caption"] = [
                "source": captionTimeline.first?.source ?? "unknown",
                "num_captions": captionTimeline.count,
            ]
        }

        guard let data = try? JSONSerialization.data(withJSONObject: output, options: .prettyPrinted) else {
            return false
        }
        return FileManager.default.createFile(atPath: outputURL.path, contents: data)
    }

    // MARK: - Mesh Overlay Drawing

    /// Draw SMPL mesh triangles on a pixel buffer with perspective projection and z-buffering.
    /// Vertices are in camera coordinates (Y-down, Z-forward) and translated by `translation`.
    /// Shared z-buffer is passed in so multiple persons can be composited correctly.
    private static func drawMeshOverlay(
        buffer: CVPixelBuffer,
        vertices: [SIMD3<Float>],
        translation: SIMD3<Float>,
        faces: [UInt32],
        focalLength: Float,
        icx: Float, icy: Float,
        width: Int, height: Int,
        zBuffer: inout [Float],
        meshColor: (UInt8, UInt8, UInt8) = (200, 200, 200)
    ) {
        let ptr = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
        let bpr = CVPixelBufferGetBytesPerRow(buffer)

        // 1. Project all vertices to 2D using perspective projection
        var projected = [(x: Float, y: Float, z: Float)]()
        projected.reserveCapacity(vertices.count)

        for v in vertices {
            let vx = v.x + translation.x
            let vy = v.y + translation.y
            let vz = v.z + translation.z
            guard vz > 0.01 else {
                projected.append((0, 0, -1))  // behind camera
                continue
            }
            let sx = focalLength * vx / vz + icx
            let sy = focalLength * vy / vz + icy
            projected.append((sx, sy, vz))
        }

        // 2. z-buffer is passed in (shared across persons)

        // Mesh color from parameter
        let meshR: UInt8 = meshColor.0
        let meshG: UInt8 = meshColor.1
        let meshB: UInt8 = meshColor.2
        let alpha: Float = 0.6  // blend factor

        // 3. Rasterize each triangle
        let numFaces = faces.count / 3
        for f in 0..<numFaces {
            let i0 = Int(faces[f * 3])
            let i1 = Int(faces[f * 3 + 1])
            let i2 = Int(faces[f * 3 + 2])

            let p0 = projected[i0]
            let p1 = projected[i1]
            let p2 = projected[i2]

            // Skip triangles with vertices behind camera
            if p0.z < 0 || p1.z < 0 || p2.z < 0 { continue }

            // Bounding box of the triangle
            let minX = max(0, Int(min(p0.x, min(p1.x, p2.x))))
            let maxX = min(width - 1, Int(max(p0.x, max(p1.x, p2.x))) + 1)
            let minY = max(0, Int(min(p0.y, min(p1.y, p2.y))))
            let maxY = min(height - 1, Int(max(p0.y, max(p1.y, p2.y))) + 1)

            if minX > maxX || minY > maxY { continue }

            // Edge function coefficients for barycentric coordinates
            let dx01 = p1.x - p0.x, dy01 = p1.y - p0.y
            let dx12 = p2.x - p1.x, dy12 = p2.y - p1.y
            let dx20 = p0.x - p2.x, dy20 = p0.y - p2.y

            let area = dx01 * dy20 - dy01 * dx20
            if abs(area) < 1e-6 { continue }  // degenerate triangle
            let invArea = 1.0 / area

            // Compute face normal for simple shading
            let v01 = SIMD3<Float>(vertices[i1].x - vertices[i0].x, vertices[i1].y - vertices[i0].y, vertices[i1].z - vertices[i0].z)
            let v02 = SIMD3<Float>(vertices[i2].x - vertices[i0].x, vertices[i2].y - vertices[i0].y, vertices[i2].z - vertices[i0].z)
            var normal = simd_cross(v01, v02)
            let nLen = simd_length(normal)
            if nLen > 1e-8 { normal /= nLen }
            // Light from camera direction (0, 0, 1) — dot product with face normal
            let lightDir = SIMD3<Float>(0, 0, -1)  // toward the camera
            let shade = max(0.2, abs(simd_dot(normal, lightDir)))

            let shadedR = UInt8(min(255, Float(meshR) * shade))
            let shadedG = UInt8(min(255, Float(meshG) * shade))
            let shadedB = UInt8(min(255, Float(meshB) * shade))

            for py in minY...maxY {
                for px in minX...maxX {
                    let fx = Float(px) + 0.5
                    let fy = Float(py) + 0.5

                    // Barycentric coordinates
                    let w0 = ((p1.x - fx) * (p2.y - fy) - (p2.x - fx) * (p1.y - fy)) * invArea
                    let w1 = ((p2.x - fx) * (p0.y - fy) - (p0.x - fx) * (p2.y - fy)) * invArea
                    let w2 = 1.0 - w0 - w1

                    if w0 >= 0 && w1 >= 0 && w2 >= 0 {
                        // Interpolate depth
                        let z = w0 * p0.z + w1 * p1.z + w2 * p2.z

                        let zIdx = py * width + px
                        if z < zBuffer[zIdx] {
                            zBuffer[zIdx] = z

                            // Alpha-blend with existing pixel
                            let off = py * bpr + px * 4
                            let bgB = Float(ptr[off])
                            let bgG = Float(ptr[off + 1])
                            let bgR = Float(ptr[off + 2])

                            ptr[off]     = UInt8(bgB * (1 - alpha) + Float(shadedB) * alpha)
                            ptr[off + 1] = UInt8(bgG * (1 - alpha) + Float(shadedG) * alpha)
                            ptr[off + 2] = UInt8(bgR * (1 - alpha) + Float(shadedR) * alpha)
                            ptr[off + 3] = 255
                        }
                    }
                }
            }
        }
    }

    // MARK: - Drawing Helpers

    private static func drawComparePanelBackground(cg: CGContext, rect: CGRect, title: String) {
        cg.setFillColor(UIColor(white: 0.13, alpha: 1).cgColor)
        cg.fill(rect)
        cg.setStrokeColor(UIColor.white.withAlphaComponent(0.2).cgColor)
        cg.setLineWidth(1)
        cg.stroke(rect)

        let titleAttr: [NSAttributedString.Key: Any] = [
            .font: UIFont.monospacedSystemFont(ofSize: 15, weight: .semibold),
            .foregroundColor: UIColor.white,
        ]
        NSString(string: title).draw(
            in: CGRect(x: rect.minX + 10, y: rect.minY + 8, width: rect.width - 20, height: 20),
            withAttributes: titleAttr
        )
    }

    private static func drawGlobalPointCloud(
        cg: CGContext,
        rect: CGRect,
        vertices: [SIMD3<Float>],
        translation: SIMD3<Float>?,
        color: UIColor
    ) {
        guard !vertices.isEmpty else { return }

        let yaw: Float = 0.55
        let pitch: Float = -0.2
        let cosY = cos(yaw), sinY = sin(yaw)
        let cosP = cos(pitch), sinP = sin(pitch)

        let focal = Float(min(rect.width, rect.height) * 0.95)
        let cx = Float(rect.midX)
        let cy = Float(rect.midY)
        let camZ: Float = 3.5

        cg.setFillColor(color.withAlphaComponent(0.8).cgColor)

        for v in vertices {
            let tx = v.x + (translation?.x ?? 0)
            let ty = v.y + (translation?.y ?? 0)
            let tz = v.z + (translation?.z ?? 0)

            let x1 = cosY * tx + sinY * tz
            let z1 = -sinY * tx + cosY * tz
            let y2 = cosP * ty - sinP * z1
            let z2 = sinP * ty + cosP * z1
            let vz = z2 + camZ
            if vz <= 0.05 { continue }

            let sx = focal * x1 / vz + cx
            let sy = -focal * y2 / vz + cy
            let px = CGFloat(sx)
            let py = CGFloat(sy)
            if !rect.insetBy(dx: 2, dy: 2).contains(CGPoint(x: px, y: py)) { continue }

            cg.fillEllipse(in: CGRect(x: px - 1.0, y: py - 1.0, width: 2.0, height: 2.0))
        }
    }

    private static func drawBenchmarkTable(
        cg: CGContext,
        rect: CGRect,
        benchmarks: [PerformanceMetrics.ModelBenchmark],
        frameIndex: Int,
        totalFrames: Int,
        smplEnabled: Bool
    ) {
        drawComparePanelBackground(cg: cg, rect: rect, title: "Benchmark")

        let headerAttr: [NSAttributedString.Key: Any] = [
            .font: UIFont.monospacedSystemFont(ofSize: 11, weight: .semibold),
            .foregroundColor: UIColor.white.withAlphaComponent(0.75),
        ]
        let rowAttr: [NSAttributedString.Key: Any] = [
            .font: UIFont.monospacedSystemFont(ofSize: 11, weight: .regular),
            .foregroundColor: UIColor.white,
        ]

        let y0 = rect.minY + 36
        NSString(string: "Model      GVHMR   SMPL   Total   ms/f    MB").draw(
            in: CGRect(x: rect.minX + 10, y: y0, width: rect.width - 20, height: 16),
            withAttributes: headerAttr
        )

        for (idx, b) in benchmarks.enumerated() {
            let model = b.model.rawValue.padding(toLength: 9, withPad: " ", startingAt: 0)
            let row = model
                + String(format: " %6.1f %6.1f %6.1f %6.0f %6.0f",
                         b.gvhmrTimeSec,
                         b.smplTimeSec,
                         b.totalTimeSec,
                         b.avgGVHMRMs,
                         b.peakMemoryMB)
            NSString(string: row).draw(
                in: CGRect(x: rect.minX + 10, y: y0 + CGFloat(18 + idx * 16), width: rect.width - 20, height: 16),
                withAttributes: rowAttr
            )
        }

        let footer = "Frame \(min(frameIndex + 1, totalFrames))/\(max(totalFrames, 1)) | SMPL: \(smplEnabled ? "on" : "off")"
        NSString(string: footer).draw(
            in: CGRect(x: rect.minX + 10, y: rect.maxY - 24, width: rect.width - 20, height: 16),
            withAttributes: headerAttr
        )
    }

    /// Draws a 3D mesh in a global view using a fixed virtual camera and z-buffering.
    private static func drawGlobalMesh(
        buffer: CVPixelBuffer,
        vertices: [SIMD3<Float>],
        translation: SIMD3<Float>? = nil,
        faces: [UInt32],
        width: Int,
        height: Int,
        zBuffer: inout [Float],
        meshColor: (UInt8, UInt8, UInt8) = (225, 235, 245)
    ) {
        let ptr = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
        let bpr = CVPixelBufferGetBytesPerRow(buffer)

        // Slight yaw and pitch to make depth visible in the global preview.
        let yaw: Float = 0.55
        let pitch: Float = -0.2
        let cosY = cos(yaw), sinY = sin(yaw)
        let cosP = cos(pitch), sinP = sin(pitch)

        let focal: Float = 420
        let cx = Float(width) / 2
        let cy = Float(height) / 2 - 20
        let camZ: Float = 3.0

        var projected = [(x: Float, y: Float, z: Float)]()
        projected.reserveCapacity(vertices.count)

        for v in vertices {
            let tx = v.x + (translation?.x ?? 0)
            let ty = v.y + (translation?.y ?? 0)
            let tz = v.z + (translation?.z ?? 0)

            let x1 = cosY * tx + sinY * tz
            let z1 = -sinY * tx + cosY * tz

            let y2 = cosP * ty - sinP * z1
            let z2 = sinP * ty + cosP * z1

            let vz = z2 + camZ
            if vz <= 0.01 {
                projected.append((0, 0, -1))
                continue
            }

            let sx = focal * x1 / vz + cx
            let sy = -focal * y2 / vz + cy
            projected.append((sx, sy, vz))
        }

        let numFaces = faces.count / 3
        for f in 0..<numFaces {
            let i0 = Int(faces[f * 3])
            let i1 = Int(faces[f * 3 + 1])
            let i2 = Int(faces[f * 3 + 2])

            let p0 = projected[i0]
            let p1 = projected[i1]
            let p2 = projected[i2]
            if p0.z < 0 || p1.z < 0 || p2.z < 0 { continue }

            let minX = max(0, Int(min(p0.x, min(p1.x, p2.x))))
            let maxX = min(width - 1, Int(max(p0.x, max(p1.x, p2.x))) + 1)
            let minY = max(0, Int(min(p0.y, min(p1.y, p2.y))))
            let maxY = min(height - 1, Int(max(p0.y, max(p1.y, p2.y))) + 1)
            if minX > maxX || minY > maxY { continue }

            let dx01 = p1.x - p0.x, dy01 = p1.y - p0.y
            let dx20 = p0.x - p2.x, dy20 = p0.y - p2.y
            let area = dx01 * dy20 - dy01 * dx20
            if abs(area) < 1e-6 { continue }
            let invArea = 1.0 / area

            let v01 = vertices[i1] - vertices[i0]
            let v02 = vertices[i2] - vertices[i0]
            var normal = simd_cross(v01, v02)
            let nLen = simd_length(normal)
            if nLen > 1e-8 { normal /= nLen }
            let lightDir = SIMD3<Float>(0, -0.2, -1)
            let shade = max(0.2, abs(simd_dot(normal, simd_normalize(lightDir))))

            let baseR: Float = Float(meshColor.0)
            let baseG: Float = Float(meshColor.1)
            let baseB: Float = Float(meshColor.2)
            let r = UInt8(min(255, baseR * shade))
            let g = UInt8(min(255, baseG * shade))
            let b = UInt8(min(255, baseB * shade))

            for py in minY...maxY {
                for px in minX...maxX {
                    let fx = Float(px) + 0.5
                    let fy = Float(py) + 0.5

                    let w0 = ((p1.x - fx) * (p2.y - fy) - (p2.x - fx) * (p1.y - fy)) * invArea
                    let w1 = ((p2.x - fx) * (p0.y - fy) - (p0.x - fx) * (p2.y - fy)) * invArea
                    let w2 = 1.0 - w0 - w1

                    if w0 >= 0 && w1 >= 0 && w2 >= 0 {
                        let z = w0 * p0.z + w1 * p1.z + w2 * p2.z
                        let zIdx = py * width + px
                        if z < zBuffer[zIdx] {
                            zBuffer[zIdx] = z
                            let off = py * bpr + px * 4
                            ptr[off] = b
                            ptr[off + 1] = g
                            ptr[off + 2] = r
                            ptr[off + 3] = 255
                        }
                    }
                }
            }
        }
    }

    /// Draw SMPL skeleton on a pixel buffer (BGRA format).
    private static func drawSkeleton(buffer: CVPixelBuffer, joints2D: [CGPoint], width: Int, height: Int) {
        let ptr = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
        let bpr = CVPixelBufferGetBytesPerRow(buffer)

        // Draw bones
        for (boneIdx, bone) in SMPLSkeleton.bones.enumerated() {
            let p1 = joints2D[bone.0]
            let p2 = joints2D[bone.1]

            let color = SMPLSkeleton.boneColors[boneIdx]
            let r = UInt8(color.0 * 255)
            let g = UInt8(color.1 * 255)
            let b = UInt8(color.2 * 255)

            drawLine(ptr: ptr, bpr: bpr, width: width, height: height,
                     x0: Int(p1.x), y0: Int(p1.y), x1: Int(p2.x), y1: Int(p2.y),
                     r: r, g: g, b: b, thickness: 2)
        }

        // Draw joints
        for (_, pt) in joints2D.enumerated() {
            drawCircle(ptr: ptr, bpr: bpr, width: width, height: height,
                       cx: Int(pt.x), cy: Int(pt.y), radius: 3,
                       r: 255, g: 255, b: 255)
        }
    }

    private static func drawLine(ptr: UnsafeMutablePointer<UInt8>, bpr: Int,
                                  width: Int, height: Int,
                                  x0: Int, y0: Int, x1: Int, y1: Int,
                                  r: UInt8, g: UInt8, b: UInt8, thickness: Int) {
        // Bresenham's line algorithm
        var x = x0, y = y0
        let dx = abs(x1 - x0), dy = abs(y1 - y0)
        let sx = x0 < x1 ? 1 : -1
        let sy = y0 < y1 ? 1 : -1
        var err = dx - dy

        let halfT = thickness / 2

        while true {
            for ty in -halfT...halfT {
                for tx in -halfT...halfT {
                    let px = x + tx, py = y + ty
                    if px >= 0 && px < width && py >= 0 && py < height {
                        let off = py * bpr + px * 4
                        ptr[off] = b
                        ptr[off + 1] = g
                        ptr[off + 2] = r
                        ptr[off + 3] = 255
                    }
                }
            }

            if x == x1 && y == y1 { break }
            let e2 = 2 * err
            if e2 > -dy { err -= dy; x += sx }
            if e2 < dx { err += dx; y += sy }
        }
    }

    private static func drawCircle(ptr: UnsafeMutablePointer<UInt8>, bpr: Int,
                                    width: Int, height: Int,
                                    cx: Int, cy: Int, radius: Int,
                                    r: UInt8, g: UInt8, b: UInt8) {
        for dy in -radius...radius {
            for dx in -radius...radius {
                if dx * dx + dy * dy <= radius * radius {
                    let px = cx + dx, py = cy + dy
                    if px >= 0 && px < width && py >= 0 && py < height {
                        let off = py * bpr + px * 4
                        ptr[off] = b
                        ptr[off + 1] = g
                        ptr[off + 2] = r
                        ptr[off + 3] = 255
                    }
                }
            }
        }
    }
}
