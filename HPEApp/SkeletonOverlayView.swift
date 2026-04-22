import SwiftUI

/// Overlay that draws the GVHMR 3D skeleton on top of the camera feed.
/// Supports both single-person and multi-person modes.
struct SkeletonOverlayView: View {
    let result: GVHMRResult?
    let inputKeypoints: [SIMD3<Float>]
    let inputBBox: CGRect
    let imageSize: CGSize
    let viewSize: CGSize
    let showInput: Bool
    let show3D: Bool
    /// Multi-person results (optional). When set, draws colored skeletons per person.
    var multiPersonResults: [PersonFrameResult]?
    var selectedTrackID: Int?

    var body: some View {
        Canvas { context, size in
            let scaleX = size.width / max(imageSize.width, 1)
            let scaleY = size.height / max(imageSize.height, 1)
            // Use max to match CameraPreviewView's .resizeAspectFill
            let scale = max(scaleX, scaleY)
            let offsetX = (size.width - imageSize.width * scale) / 2
            let offsetY = (size.height - imageSize.height * scale) / 2

            // Helper: convert image coords to view coords
            func toView(_ point: CGPoint) -> CGPoint {
                return CGPoint(
                    x: point.x * scale + offsetX,
                    y: point.y * scale + offsetY
                )
            }

            // Multi-person mode
            if let persons = multiPersonResults, !persons.isEmpty {
                for person in persons {
                    let rgb = PersonColors.color(for: person.trackID)
                    let color = Color(red: Double(rgb.0), green: Double(rgb.1), blue: Double(rgb.2))
                    let isSelected = selectedTrackID == person.trackID

                    if showInput {
                        drawInputPose(context: context, keypoints: person.keypoints,
                                      bbox: person.bbox, scale: scale,
                                      offsetX: offsetX, offsetY: offsetY,
                                      tintColor: color,
                                      isSelected: isSelected)
                    }

                    if show3D {
                        drawSMPLSkeleton(context: context, result: person.gvhmrResult,
                                         toView: toView, tintColor: color)
                    }
                }
            } else {
                // Single-person mode (backward compatible)
                if showInput {
                    drawInputPose(context: context, keypoints: inputKeypoints,
                                  bbox: inputBBox, scale: scale,
                                  offsetX: offsetX, offsetY: offsetY)
                }

                if show3D, let result = result {
                    drawSMPLSkeleton(context: context, result: result, toView: toView)
                }
            }
        }
    }

    // MARK: - Draw Input 2D Pose

    private func drawInputPose(
        context: GraphicsContext, keypoints: [SIMD3<Float>],
        bbox: CGRect, scale: CGFloat, offsetX: CGFloat, offsetY: CGFloat,
        tintColor: Color? = nil,
        isSelected: Bool = false
    ) {
        let boneColor = tintColor ?? .cyan
        let boxColor = tintColor ?? .yellow

        // Draw bounding box
        let bboxView = CGRect(
            x: bbox.origin.x * scale + offsetX,
            y: bbox.origin.y * scale + offsetY,
            width: bbox.width * scale,
            height: bbox.height * scale
        )
        context.stroke(
            Path(roundedRect: bboxView, cornerRadius: 4),
            with: .color((isSelected ? Color.white : boxColor).opacity(isSelected ? 0.9 : 0.6)),
            lineWidth: isSelected ? 3 : 2
        )

        // Draw COCO-17 skeleton connections
        let cocoConnections: [(Int, Int)] = [
            (0, 1), (0, 2), (1, 3), (2, 4),       // face
            (5, 6),                                  // shoulders
            (5, 7), (7, 9),                          // left arm
            (6, 8), (8, 10),                         // right arm
            (5, 11), (6, 12),                        // torso
            (11, 12),                                // hips
            (11, 13), (13, 15),                      // left leg
            (12, 14), (14, 16),                      // right leg
        ]

        for (i, j) in cocoConnections {
            guard i < keypoints.count, j < keypoints.count else { continue }
            let kpi = keypoints[i]
            let kpj = keypoints[j]
            guard kpi.z > 0.3 && kpj.z > 0.3 else { continue }

            let p1 = CGPoint(x: CGFloat(kpi.x) * scale + offsetX,
                             y: CGFloat(kpi.y) * scale + offsetY)
            let p2 = CGPoint(x: CGFloat(kpj.x) * scale + offsetX,
                             y: CGFloat(kpj.y) * scale + offsetY)

            var path = Path()
            path.move(to: p1)
            path.addLine(to: p2)
            context.stroke(path, with: .color(boneColor.opacity(0.5)), lineWidth: 2)
        }

        // Draw keypoint dots
        for kp in keypoints where kp.z > 0.3 {
            let pt = CGPoint(x: CGFloat(kp.x) * scale + offsetX,
                             y: CGFloat(kp.y) * scale + offsetY)
            let circle = Path(ellipseIn: CGRect(x: pt.x - 3, y: pt.y - 3, width: 6, height: 6))
            context.fill(circle, with: .color(boneColor))
        }
    }

    // MARK: - Draw SMPL 3D Skeleton

    private func drawSMPLSkeleton(
        context: GraphicsContext,
        result: GVHMRResult,
        toView: (CGPoint) -> CGPoint,
        tintColor: Color? = nil
    ) {
        let joints2D = result.joints2D

        // Draw bones
        for (idx, bone) in SMPLSkeleton.bones.enumerated() {
            guard bone.0 < joints2D.count, bone.1 < joints2D.count else { continue }
            let p1 = toView(joints2D[bone.0])
            let p2 = toView(joints2D[bone.1])

            let color: Color
            if let tint = tintColor {
                color = tint
            } else {
                let c = SMPLSkeleton.boneColors[idx]
                color = Color(red: Double(c.0), green: Double(c.1), blue: Double(c.2))
            }

            var path = Path()
            path.move(to: p1)
            path.addLine(to: p2)
            context.stroke(path, with: .color(color.opacity(0.9)),
                           style: StrokeStyle(lineWidth: 4, lineCap: .round))
        }

        // Draw joints
        for (i, pt) in joints2D.enumerated() {
            let viewPt = toView(pt)
            let radius: CGFloat = i == 0 ? 6 : 4  // larger pelvis
            let circle = Path(ellipseIn: CGRect(
                x: viewPt.x - radius, y: viewPt.y - radius,
                width: radius * 2, height: radius * 2
            ))

            let jointColor: Color
            if let tint = tintColor {
                jointColor = tint
            } else {
                let isLeft = SMPLSkeleton.jointNames[i].hasPrefix("L_")
                let isRight = SMPLSkeleton.jointNames[i].hasPrefix("R_")
                jointColor = isLeft ? .blue : (isRight ? .red : .white)
            }
            context.fill(circle, with: .color(jointColor))
            context.stroke(circle, with: .color(.white.opacity(0.8)), lineWidth: 1)
        }
    }
}
