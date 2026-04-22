import Foundation
import simd

/// Lightweight semantic caption generated from GVHMR outputs.
///
/// This is intentionally cheap enough for on-device use and acts as a bridge
/// between geometric pose output and language output for Caption-style UX.
struct CaptionSemanticFrame: Identifiable {
    let id = UUID()
    let frameIndex: Int
    let timestampSec: Double
    let caption: String
    let confidence: Float
    let source: String
    let actionTag: String
    let postureTag: String
    let raisedLimbs: [String]
    let occludedParts: [String]
    let attributes: [String]
}

/// Pose-to-language fusion engine.
///
/// The engine produces motion-aware captions from GVHMR pose streams.
/// It is deterministic and fast, and can later be replaced with a true visual
/// caption model while preserving the same API surface.
final class CaptionFusionEngine {

    private var previousPelvis: SIMD3<Float>?
    private var previousTimestampSec: Double?
    private var previousLeftWrist: SIMD3<Float>?
    private var previousRightWrist: SIMD3<Float>?
    private var previousLeftAnkle: SIMD3<Float>?
    private var previousRightAnkle: SIMD3<Float>?
    private var previousBodyYaw: Float?
    private var previousPelvis2D: CGPoint?
    private var smoothedSpeed: Float = 0
    private var smoothedLimbMotion: Float = 0
    private var smoothedYawRate: Float = 0

    private let fallbackFPS: Float = 30
    private let sourceName = "pose-caption-fusion-v4"

    func reset() {
        previousPelvis = nil
        previousTimestampSec = nil
        previousLeftWrist = nil
        previousRightWrist = nil
        previousLeftAnkle = nil
        previousRightAnkle = nil
        previousBodyYaw = nil
        previousPelvis2D = nil
        smoothedSpeed = 0
        smoothedLimbMotion = 0
        smoothedYawRate = 0
    }

    func analyze(
        result: GVHMRResult,
        frameIndex: Int,
        timestampSec: Double,
        personCount: Int,
        visualCaption: String? = nil
    ) -> CaptionSemanticFrame {
        if result.joints3D.count < 16 {
            previousPelvis = nil
            previousTimestampSec = nil
            let caption = personCount > 1
                ? "\(personCount) people visible; selected subject not confidently tracked."
                : "Subject not confidently tracked."
            return CaptionSemanticFrame(
                frameIndex: frameIndex,
                timestampSec: timestampSec,
                caption: caption,
                confidence: 0.35,
                source: sourceName,
                actionTag: "untracked",
                postureTag: "unknown",
                raisedLimbs: [],
                occludedParts: ["full_body"],
                attributes: ["tracking_low_confidence"]
            )
        }

        let pelvis = joint(result.joints3D, 0)
        let pelvis2D = joint2D(result.joints2D, 0)
        let head = joint(result.joints3D, 15)
        let leftWrist = joint(result.joints3D, 20)
        let rightWrist = joint(result.joints3D, 21)
        let leftHip = joint(result.joints3D, 1)
        let rightHip = joint(result.joints3D, 2)
        let leftKnee = joint(result.joints3D, 4)
        let rightKnee = joint(result.joints3D, 5)
        let leftAnkle = joint(result.joints3D, 7)
        let rightAnkle = joint(result.joints3D, 8)

        let dt = timeDelta(nowSec: timestampSec)
        let velocity = computeVelocity(currentPelvis: pelvis, dt: dt)
        let imageLateralVelocity = computeImageLateralVelocity(currentPelvis2D: pelvis2D, dt: dt)
        let speed = simd_length(velocity)
        let limbMotion = computeLimbMotion(
            leftWrist: leftWrist,
            rightWrist: rightWrist,
            leftAnkle: leftAnkle,
            rightAnkle: rightAnkle,
            dt: dt
        )
        let yawRate = computeYawRate(
            leftHip: leftHip,
            rightHip: rightHip,
            dt: dt
        )

        // Stabilize noisy per-frame velocities while still reacting to motion.
        smoothedSpeed = 0.65 * smoothedSpeed + 0.35 * speed
        smoothedLimbMotion = 0.6 * smoothedLimbMotion + 0.4 * limbMotion
        smoothedYawRate = 0.7 * smoothedYawRate + 0.3 * abs(yawRate)

        let raisedLimbs = detectRaisedLimbs(
            head: head,
            leftWrist: leftWrist,
            rightWrist: rightWrist,
            leftHip: leftHip,
            rightHip: rightHip,
            leftKnee: leftKnee,
            rightKnee: rightKnee,
            leftAnkle: leftAnkle,
            rightAnkle: rightAnkle
        )

        let postureTag = classifyPosture(
            pelvis: pelvis,
            head: head,
            leftHip: leftHip,
            rightHip: rightHip,
            leftKnee: leftKnee,
            rightKnee: rightKnee,
            leftAnkle: leftAnkle,
            rightAnkle: rightAnkle
        )

        let occludedParts = estimateOcclusion(
            joints3D: result.joints3D,
            joints2D: result.joints2D
        )

        let (baseCaption, actionTag, confidence) = classifyAction(
            speed: smoothedSpeed,
            limbMotion: smoothedLimbMotion,
            yawRate: smoothedYawRate,
            imageLateralVelocity: imageLateralVelocity,
            velocity: velocity,
            postureTag: postureTag,
            raisedLimbs: raisedLimbs
        )

        let motionCaption: String
        if personCount > 1 {
            motionCaption = "\(personCount) people visible; selected subject \(baseCaption)."
        } else {
            motionCaption = "Subject \(baseCaption)."
        }

        let caption: String
        let motionInfo = buildMotionInfo(
            baseCaption: baseCaption,
            postureTag: postureTag,
            raisedLimbs: raisedLimbs,
            occludedParts: occludedParts
        )
        if let visualCaption,
           !visualCaption.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            caption = "\(motionCaption) \(motionInfo) Visual: \(visualCaption)."
        } else {
            caption = "\(motionCaption) \(motionInfo)"
        }

        var attributes = [String]()
        attributes.append("people_visible_\(max(personCount, 1))")
        attributes.append("posture_\(postureTag)")
        attributes.append("action_\(actionTag)")
        if !raisedLimbs.isEmpty {
            attributes.append("raised_\(raisedLimbs.joined(separator: "_"))")
        }
        if !occludedParts.isEmpty {
            attributes.append("occluded_\(occludedParts.joined(separator: "_"))")
        }

        previousPelvis = pelvis
        previousTimestampSec = timestampSec
        previousLeftWrist = leftWrist
        previousRightWrist = rightWrist
        previousLeftAnkle = leftAnkle
        previousRightAnkle = rightAnkle

        return CaptionSemanticFrame(
            frameIndex: frameIndex,
            timestampSec: timestampSec,
            caption: caption,
            confidence: confidence,
            source: sourceName,
            actionTag: actionTag,
            postureTag: postureTag,
            raisedLimbs: raisedLimbs,
            occludedParts: occludedParts,
            attributes: attributes
        )
    }

    func summarize(_ timeline: [CaptionSemanticFrame]) -> String {
        guard !timeline.isEmpty else {
            return "No semantic captions generated."
        }

        var counts: [String: Int] = [:]
        var confidenceSum: Float = 0
        for item in timeline {
            counts[item.actionTag, default: 0] += 1
            confidenceSum += item.confidence
        }

        let dominant = counts.max(by: { $0.value < $1.value })?.key ?? "unknown"
        let avgConfidence = confidenceSum / Float(timeline.count)
        let duration = timeline.last?.timestampSec ?? 0

        return String(
            format: "Dominant action: %@ | Avg confidence: %.2f | Duration: %.1fs",
            dominant,
            avgConfidence,
            duration
        )
    }

    // MARK: - Internal

    private func classifyAction(
        speed: Float,
        limbMotion: Float,
        yawRate: Float,
        imageLateralVelocity: Float,
        velocity: SIMD3<Float>,
        postureTag: String,
        raisedLimbs: [String]
    ) -> (caption: String, actionTag: String, confidence: Float) {
        let leftLegRaised = raisedLimbs.contains("left_leg")
        let rightLegRaised = raisedLimbs.contains("right_leg")
        let singleLegRaised = leftLegRaised != rightLegRaised
        let verticalSpeed = abs(velocity.y)

        if postureTag == "lying" && verticalSpeed > 0.95 && limbMotion > 0.75 {
            return ("is flipping", "flipping", 0.83)
        }
        if yawRate > 2.8 && speed > 0.18 {
            return ("is spinning", "spinning", 0.86)
        }
        if postureTag == "lying" {
            return ("is lying down", "lying", 0.90)
        }
        if postureTag == "sitting" {
            return ("is sitting", "sitting", 0.88)
        }
        if postureTag == "squatting" {
            return ("is doing a squat", "squat", 0.90)
        }
        if postureTag == "bending" {
            return ("appears to be bending over", "bending", 0.83)
        }
        if raisedLimbs.contains("left_hand") && raisedLimbs.contains("right_hand") {
            return ("is raising both hands", "hands_up", 0.90)
        }
        if raisedLimbs.contains("left_hand") || raisedLimbs.contains("right_hand") {
            return ("is raising one hand", "one_hand_up", 0.84)
        }

        let direction = movementDirection(imageLateralVelocity)

        if singleLegRaised && verticalSpeed > 0.75 && limbMotion > 0.6 {
            return ("is doing a single leg hop", "single_leg_hop", 0.88)
        }
        if verticalSpeed > 0.85 && limbMotion > 0.55 {
            return ("is jumping", "jumping", 0.90)
        }
        if speed > 0.9 || (speed > 0.65 && limbMotion > 0.55) {
            return ("is running \(direction)", "running", 0.90)
        }
        if limbMotion > 0.75 && speed > 0.2 {
            return ("is dancing with active limb motion", "dancing", 0.86)
        }
        if speed > 0.18 || limbMotion > 0.45 {
            return ("is moving \(direction)", "walking", 0.80)
        }
        return ("is mostly stationary", "standing", 0.76)
    }

    private func computeLimbMotion(
        leftWrist: SIMD3<Float>,
        rightWrist: SIMD3<Float>,
        leftAnkle: SIMD3<Float>,
        rightAnkle: SIMD3<Float>,
        dt: Float
    ) -> Float {
        let safeDt = max(dt, 1e-4)

        var speeds = [Float]()
        if let prev = previousLeftWrist { speeds.append(simd_length(leftWrist - prev) / safeDt) }
        if let prev = previousRightWrist { speeds.append(simd_length(rightWrist - prev) / safeDt) }
        if let prev = previousLeftAnkle { speeds.append(simd_length(leftAnkle - prev) / safeDt) }
        if let prev = previousRightAnkle { speeds.append(simd_length(rightAnkle - prev) / safeDt) }

        guard !speeds.isEmpty else { return 0 }
        return speeds.reduce(0, +) / Float(speeds.count)
    }

    private func computeYawRate(
        leftHip: SIMD3<Float>,
        rightHip: SIMD3<Float>,
        dt: Float
    ) -> Float {
        let hipVec = rightHip - leftHip
        let horizontal = SIMD2<Float>(hipVec.x, hipVec.z)
        let mag = simd_length(horizontal)
        guard mag > 1e-4 else { return 0 }

        let yaw = atan2(horizontal.y, horizontal.x)
        defer { previousBodyYaw = yaw }

        guard let prevYaw = previousBodyYaw else { return 0 }
        let delta = shortestAngleDelta(from: prevYaw, to: yaw)
        return delta / max(dt, 1e-4)
    }

    private func shortestAngleDelta(from: Float, to: Float) -> Float {
        var d = to - from
        let twoPi = Float.pi * 2
        while d > Float.pi { d -= twoPi }
        while d < -Float.pi { d += twoPi }
        return d
    }

    private func classifyPosture(
        pelvis: SIMD3<Float>,
        head: SIMD3<Float>,
        leftHip: SIMD3<Float>,
        rightHip: SIMD3<Float>,
        leftKnee: SIMD3<Float>,
        rightKnee: SIMD3<Float>,
        leftAnkle: SIMD3<Float>,
        rightAnkle: SIMD3<Float>
    ) -> String {
        let torso = simd_normalize(head - pelvis)
        let vertical = SIMD3<Float>(0, -1, 0)
        let uprightScore = abs(simd_dot(torso, vertical))

        let hipY = 0.5 * (leftHip.y + rightHip.y)
        let kneeY = 0.5 * (leftKnee.y + rightKnee.y)
        let ankleY = 0.5 * (leftAnkle.y + rightAnkle.y)

        let hipToKnee = abs(kneeY - hipY)
        let hipToAnkle = max(abs(ankleY - hipY), 1e-3)
        let compression = hipToKnee / hipToAnkle

        let kneeSpread = 0.5 * (simd_length(leftKnee - leftHip) + simd_length(rightKnee - rightHip))
        let shinLength = 0.5 * (simd_length(leftAnkle - leftKnee) + simd_length(rightAnkle - rightKnee))
        let sitLike = kneeSpread > 0.15 && shinLength < 0.45

        if uprightScore < 0.42 {
            return "lying"
        }
        if uprightScore < 0.68 {
            return "bending"
        }
        if compression < 0.35 {
            return sitLike ? "sitting" : "squatting"
        }
        return "upright"
    }

    private func detectRaisedLimbs(
        head: SIMD3<Float>,
        leftWrist: SIMD3<Float>,
        rightWrist: SIMD3<Float>,
        leftHip: SIMD3<Float>,
        rightHip: SIMD3<Float>,
        leftKnee: SIMD3<Float>,
        rightKnee: SIMD3<Float>,
        leftAnkle: SIMD3<Float>,
        rightAnkle: SIMD3<Float>
    ) -> [String] {
        var raised = [String]()
        if leftWrist.y < head.y - 0.03 { raised.append("left_hand") }
        if rightWrist.y < head.y - 0.03 { raised.append("right_hand") }

        let leftLegRaise = leftAnkle.y < leftHip.y - 0.02 && simd_length(leftAnkle - leftKnee) > 0.22
        let rightLegRaise = rightAnkle.y < rightHip.y - 0.02 && simd_length(rightAnkle - rightKnee) > 0.22
        if leftLegRaise { raised.append("left_leg") }
        if rightLegRaise { raised.append("right_leg") }
        return raised
    }

    private func estimateOcclusion(joints3D: [SIMD3<Float>], joints2D: [CGPoint]) -> [String] {
        guard joints3D.count >= 22, joints2D.count >= 22 else { return ["unknown"] }

        var occluded = [String]()

        let leftWrist2D = joints2D[20]
        let rightWrist2D = joints2D[21]
        let neck2D = joints2D[12]
        if distance2D(leftWrist2D, neck2D) < 34, joints3D[20].z > joints3D[12].z + 0.08 {
            occluded.append("left_hand")
        }
        if distance2D(rightWrist2D, neck2D) < 34, joints3D[21].z > joints3D[12].z + 0.08 {
            occluded.append("right_hand")
        }

        let leftAnkle2D = joints2D[7]
        let rightAnkle2D = joints2D[8]
        let pelvis2D = joints2D[0]
        if distance2D(leftAnkle2D, pelvis2D) < 40, joints3D[7].z > joints3D[0].z + 0.12 {
            occluded.append("left_leg")
        }
        if distance2D(rightAnkle2D, pelvis2D) < 40, joints3D[8].z > joints3D[0].z + 0.12 {
            occluded.append("right_leg")
        }

        return Array(Set(occluded)).sorted()
    }

    private func buildMotionInfo(
        baseCaption: String,
        postureTag: String,
        raisedLimbs: [String],
        occludedParts: [String]
    ) -> String {
        var pieces = [String]()
        pieces.append("Action: \(baseCaption).")

        if postureTag != "upright" {
            pieces.append("Posture: \(postureTag).")
        }
        if !raisedLimbs.isEmpty {
            pieces.append("Raised limbs: \(describeRaisedOrOccludedParts(raisedLimbs, raised: true)).")
        }
        if !occludedParts.isEmpty {
            pieces.append("Occluded parts: \(describeRaisedOrOccludedParts(occludedParts, raised: false)).")
        }

        return pieces.joined(separator: " ")
    }

    private func describeRaisedOrOccludedParts(_ parts: [String], raised: Bool) -> String {
        parts.map { part in
            switch part {
            case "left_hand":
                return raised ? "left arm raised" : "left arm occluded"
            case "right_hand":
                return raised ? "right arm raised" : "right arm occluded"
            case "left_leg":
                return raised ? "left leg raised" : "left leg occluded"
            case "right_leg":
                return raised ? "right leg raised" : "right leg occluded"
            default:
                return part.replacingOccurrences(of: "_", with: " ")
            }
        }
        .joined(separator: ", ")
    }

    private func distance2D(_ a: CGPoint, _ b: CGPoint) -> CGFloat {
        let dx = a.x - b.x
        let dy = a.y - b.y
        return sqrt(dx * dx + dy * dy)
    }

    private func movementDirection(_ lateralVelocity: Float) -> String {
        let deadZone: Float = 1.5
        if abs(lateralVelocity) < deadZone {
            return ""
        }
        return lateralVelocity > 0 ? "to camera-right" : "to camera-left"
    }

    private func computeImageLateralVelocity(currentPelvis2D: CGPoint, dt: Float) -> Float {
        guard let previousPelvis2D else {
            self.previousPelvis2D = currentPelvis2D
            return 0
        }

        let deltaX = Float(currentPelvis2D.x - previousPelvis2D.x)
        self.previousPelvis2D = currentPelvis2D
        return deltaX / max(dt, 1e-4)
    }

    private func computeVelocity(currentPelvis: SIMD3<Float>, dt: Float) -> SIMD3<Float> {
        guard let previousPelvis else {
            return .zero
        }
        return (currentPelvis - previousPelvis) / max(dt, 1e-4)
    }

    private func timeDelta(nowSec: Double) -> Float {
        guard let previousTimestampSec else {
            return 1.0 / fallbackFPS
        }
        return Float(max(nowSec - previousTimestampSec, 1.0 / Double(fallbackFPS)))
    }

    private func joint(_ joints: [SIMD3<Float>], _ index: Int) -> SIMD3<Float> {
        guard joints.indices.contains(index) else {
            return .zero
        }
        return joints[index]
    }

    private func joint2D(_ joints: [CGPoint], _ index: Int) -> CGPoint {
        guard joints.indices.contains(index) else {
            return .zero
        }
        return joints[index]
    }
}
