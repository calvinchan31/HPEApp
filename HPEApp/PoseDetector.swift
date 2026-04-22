import Vision
import UIKit

/// Uses Apple Vision to detect human body pose and bounding box.
/// Returns COCO-17 format keypoints: (x_pixel, y_pixel, confidence).
class PoseDetector {

    /// Apple Vision keypoint names in COCO-17 order.
    private let cocoKeys: [VNHumanBodyPoseObservation.JointName] = [
        .nose,
        .leftEye,  .rightEye,
        .leftEar,  .rightEar,
        .leftShoulder, .rightShoulder,
        .leftElbow, .rightElbow,
        .leftWrist, .rightWrist,
        .leftHip,   .rightHip,
        .leftKnee,  .rightKnee,
        .leftAnkle, .rightAnkle,
    ]

    // MARK: - Temporal Tracking State

    /// Previous frame's bounding box for IoU-based tracking.
    private var previousBBox: CGRect?
    /// Number of consecutive frames where detection didn't match previous bbox.
    private var holdFrameCount: Int = 0
    /// Maximum frames to hold previous bbox when detection jumps.
    private let maxHoldFrames = 15

    /// Reset tracking state. Call at start of a new video or session.
    func resetTracking() {
        previousBBox = nil
        holdFrameCount = 0
    }

    /// Detect body pose in a pixel buffer.
    /// - Parameters:
    ///   - pixelBuffer: camera frame
    ///   - imageSize: frame dimensions
    /// - Returns: tuple of (17 keypoints as SIMD3, bounding box in pixels), or nil
    func detect(pixelBuffer: CVPixelBuffer, imageSize: CGSize) -> (keypoints: [SIMD3<Float>], bbox: CGRect)? {
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])

        let poseRequest = VNDetectHumanBodyPoseRequest()
        let rectRequest = VNDetectHumanRectanglesRequest()
        rectRequest.upperBodyOnly = false

        do {
            try handler.perform([poseRequest, rectRequest])
        } catch {
            return nil
        }

        // Get first detected body pose
        guard let poseObs = poseRequest.results?.first else {
            return nil
        }

        // Extract keypoints in pixel coordinates
        var keypoints = [SIMD3<Float>]()
        keypoints.reserveCapacity(17)

        for jointName in cocoKeys {
            if let point = try? poseObs.recognizedPoint(jointName),
               point.confidence > 0.1 {
                // Vision coordinates: (0,0) = bottom-left, normalized [0,1]
                // Convert to pixel coordinates with origin at top-left
                let x = Float(point.location.x * imageSize.width)
                let y = Float((1 - point.location.y) * imageSize.height)
                let conf = Float(point.confidence)
                keypoints.append(SIMD3<Float>(x, y, conf))
            } else {
                keypoints.append(SIMD3<Float>(0, 0, 0))
            }
        }

        // Get bounding box with IoU-based temporal tracking
        let bbox: CGRect
        if let tracked = selectBestRect(from: rectRequest.results, imageSize: imageSize) {
            bbox = tracked
        } else {
            // Fallback: compute bbox from keypoints
            bbox = computeBBoxFromKeypoints(keypoints, imageSize: imageSize)
        }

        return (keypoints, bbox)
    }

    /// Normalize keypoints relative to the bounding box.
    /// Matches GVHMR's normalize_kp2d: normalized_obs_xy = 2 * (obs_xy - center) / scale
    /// Uses the bbx_xys (center_x, center_y, size) with enlarge + aspect ratio already applied.
    func normalizeKeypoints(_ keypoints: [SIMD3<Float>], bbxXYS: SIMD3<Float>) -> [SIMD3<Float>] {
        let cx = bbxXYS.x
        let cy = bbxXYS.y
        let size = bbxXYS.z

        guard size > 1 else { return keypoints }

        let halfSize = size / 2
        // Boundary check uses the enlarged square bbox (center ± size/2)
        let xMin = cx - halfSize
        let xMax = cx + halfSize
        let yMin = cy - halfSize
        let yMax = cy + halfSize

        return keypoints.map { kp in
            // Normalize xy for ALL keypoints (matching Python behavior)
            let nx = (kp.x - cx) / halfSize
            let ny = (kp.y - cy) / halfSize

            var conf = kp.z
            // Zero confidence for low-confidence keypoints
            if conf < 0.5 {
                conf = 0
            }
            // Zero confidence for keypoints outside the enlarged bbox
            if kp.x < xMin || kp.x > xMax || kp.y < yMin || kp.y > yMax {
                conf = 0
            }
            return SIMD3<Float>(nx, ny, conf)
        }
    }

    // MARK: - Helpers

    /// Select the best person rectangle using IoU-based temporal tracking.
    /// Prefers detections that overlap with the previous frame's bbox to avoid
    /// jumping to a different person or artifact.
    private func selectBestRect(from results: [VNHumanObservation]?, imageSize: CGSize) -> CGRect? {
        guard let results = results, !results.isEmpty else { return nil }

        // Convert all results to pixel coordinates
        let candidates = results.map { obs -> CGRect in
            let nb = obs.boundingBox
            return CGRect(
                x: nb.origin.x * imageSize.width,
                y: (1 - nb.origin.y - nb.height) * imageSize.height,
                width: nb.width * imageSize.width,
                height: nb.height * imageSize.height
            )
        }

        let chosen: CGRect
        if let prev = previousBBox {
            // Pick candidate with highest IoU to previous bbox
            var bestIoU: Float = -1
            var bestCandidate = candidates[0]
            for cand in candidates {
                let iou = computeIoU(prev, cand)
                if iou > bestIoU {
                    bestIoU = iou
                    bestCandidate = cand
                }
            }

            if bestIoU >= 0.2 {
                // Good match — use this detection
                holdFrameCount = 0
                chosen = bestCandidate
            } else {
                // Detection jumped to a different target — hold previous bbox
                holdFrameCount += 1
                if holdFrameCount <= maxHoldFrames {
                    chosen = prev
                } else {
                    // Held too long — accept largest detection
                    holdFrameCount = 0
                    chosen = candidates.max(by: { $0.width * $0.height < $1.width * $1.height })!
                }
            }
        } else {
            // No previous — pick largest detection
            chosen = candidates.max(by: { $0.width * $0.height < $1.width * $1.height })!
            holdFrameCount = 0
        }

        previousBBox = chosen
        return chosen
    }

    private func computeIoU(_ a: CGRect, _ b: CGRect) -> Float {
        let intersection = a.intersection(b)
        if intersection.isNull { return 0 }
        let interArea = intersection.width * intersection.height
        let unionArea = a.width * a.height + b.width * b.height - interArea
        guard unionArea > 0 else { return 0 }
        return Float(interArea / unionArea)
    }

    private func computeBBoxFromKeypoints(_ keypoints: [SIMD3<Float>], imageSize: CGSize) -> CGRect {
        var minX = Float.greatestFiniteMagnitude
        var minY = Float.greatestFiniteMagnitude
        var maxX = -Float.greatestFiniteMagnitude
        var maxY = -Float.greatestFiniteMagnitude
        var found = false

        for kp in keypoints where kp.z > 0.1 {
            minX = min(minX, kp.x)
            minY = min(minY, kp.y)
            maxX = max(maxX, kp.x)
            maxY = max(maxY, kp.y)
            found = true
        }

        guard found else {
            return CGRect(x: 0, y: 0, width: imageSize.width, height: imageSize.height)
        }

        // Add padding (20%)
        let w = maxX - minX
        let h = maxY - minY
        let pad = max(w, h) * 0.2
        return CGRect(
            x: CGFloat(minX - pad),
            y: CGFloat(minY - pad),
            width: CGFloat(w + 2 * pad),
            height: CGFloat(h + 2 * pad)
        )
    }

    /// Compute the bbx_xys format: (center_x, center_y, bbox_size).
    /// Matches Python's get_bbx_xys_from_xyxy with base_enlarge=1.2 and 192:256 aspect ratio.
    func computeBBXXYS(bbox: CGRect) -> SIMD3<Float> {
        let cx = Float(bbox.midX)
        let cy = Float(bbox.midY)
        var w = Float(bbox.width)
        var h = Float(bbox.height)

        // Aspect ratio correction: fit to 192:256 (0.75) ratio
        let aspectRatio: Float = 192.0 / 256.0  // = 0.75
        if w > aspectRatio * h {
            h = w / aspectRatio
        } else if w < aspectRatio * h {
            w = h * aspectRatio
        }

        // base_enlarge = 1.2
        let size = max(h, w) * 1.2
        return SIMD3<Float>(cx, cy, size)
    }
}
