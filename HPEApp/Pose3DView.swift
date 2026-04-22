import SwiftUI

/// Standalone 3D skeleton visualization rendered from a fixed viewpoint.
/// Shows the GVHMR-estimated pose on a dark background beside the camera feed.
struct Pose3DView: View {
    let result: GVHMRResult?
    var debugMessage: String? = nil
    @State private var rotationAngle: Float = 0    // Y-axis rotation in radians
    @State private var isDragging = false

    var body: some View {
        GeometryReader { geo in
            ZStack {
                // Dark gradient background
                LinearGradient(
                    colors: [Color(white: 0.08), Color(white: 0.15)],
                    startPoint: .top, endPoint: .bottom
                )

                // Grid floor
                Canvas { context, size in
                    drawGrid(context: context, size: size)
                }

                // 3D skeleton
                Canvas { context, size in
                    guard let result = result else {
                        drawPlaceholder(context: context, size: size)
                        return
                    }
                    drawSkeleton3D(context: context, size: size, result: result)
                }

                // Label
                VStack {
                    HStack {
                        Text("3D Pose")
                            .font(.caption2)
                            .fontWeight(.semibold)
                            .foregroundColor(.white.opacity(0.7))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 3)
                            .background(.ultraThinMaterial)
                            .clipShape(Capsule())
                        Spacer()
                    }
                    .padding(8)
                    Spacer()
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 12))
            .gesture(
                DragGesture()
                    .onChanged { value in
                        rotationAngle += Float(value.translation.width) * 0.005
                    }
            )
        }
    }

    // MARK: - Grid Drawing

    private func drawGrid(context: GraphicsContext, size: CGSize) {
        let centerX = size.width / 2
        let groundY = size.height * 0.82
        let gridColor = Color.white.opacity(0.08)
        let lineCount = 9
        let spacing: CGFloat = size.width / CGFloat(lineCount + 1)

        // Horizontal lines with perspective
        for i in 0..<5 {
            let y = groundY + CGFloat(i) * 8
            let perspectiveScale = 1.0 - CGFloat(i) * 0.05
            let halfWidth = (size.width / 2) * perspectiveScale
            var path = Path()
            path.move(to: CGPoint(x: centerX - halfWidth, y: y))
            path.addLine(to: CGPoint(x: centerX + halfWidth, y: y))
            context.stroke(path, with: .color(gridColor), lineWidth: 0.5)
        }

        // Vertical lines converging to vanishing point
        for i in 0...lineCount {
            let x = CGFloat(i) * spacing
            var path = Path()
            path.move(to: CGPoint(x: x, y: groundY))
            let topX = centerX + (x - centerX) * 0.7
            path.addLine(to: CGPoint(x: topX, y: groundY + 40))
            context.stroke(path, with: .color(gridColor), lineWidth: 0.5)
        }
    }

    // MARK: - Placeholder

    private func drawPlaceholder(context: GraphicsContext, size: CGSize) {
        let center = CGPoint(x: size.width / 2, y: size.height / 2)
        let msg = debugMessage ?? "Waiting for pose..."
        let text = Text(msg)
            .font(.caption)
            .foregroundColor(.white.opacity(0.5))
        context.draw(text, at: center)
    }

    // MARK: - 3D Skeleton Drawing

    private func drawSkeleton3D(context: GraphicsContext, size: CGSize, result: GVHMRResult) {
        let joints3D = result.joints3D
        guard joints3D.count == 22 else { return }

        // Apply Y-axis rotation for interactive viewing
        let cosA = cos(rotationAngle)
        let sinA = sin(rotationAngle)

        // Center the skeleton by computing bounding box
        var minP = joints3D[0]
        var maxP = joints3D[0]
        for j in joints3D {
            minP = SIMD3<Float>(min(minP.x, j.x), min(minP.y, j.y), min(minP.z, j.z))
            maxP = SIMD3<Float>(max(maxP.x, j.x), max(maxP.y, j.y), max(maxP.z, j.z))
        }
        let center3D = (minP + maxP) / 2
        let extent = max(maxP.x - minP.x, maxP.y - minP.y, maxP.z - minP.z, 0.01)

        // Project each joint: rotate around Y, then orthographic projection scaled to view
        let viewScale = Float(min(size.width, size.height)) * 0.7 / extent
        let viewCenterX = Float(size.width) / 2
        let viewCenterY = Float(size.height) * 0.45 // slightly above center for aesthetics

        var projected = [(CGPoint, Float)]()  // (2D point, depth for sizing)
        projected.reserveCapacity(22)

        for joint in joints3D {
            let p = joint - center3D

            // Y-axis rotation
            let rx = p.x * cosA + p.z * sinA
            let ry = p.y
            let rz = -p.x * sinA + p.z * cosA

            // Orthographic projection: camera Y-down matches screen Y-down, no negation needed
            let screenX = CGFloat(rx * viewScale + viewCenterX)
            let screenY = CGFloat(ry * viewScale + viewCenterY)

            projected.append((CGPoint(x: screenX, y: screenY), rz))
        }

        // Sort bones by average depth (back to front) for painter's algorithm
        let boneIndices = SMPLSkeleton.bones.indices.sorted { a, b in
            let boneA = SMPLSkeleton.bones[a]
            let boneB = SMPLSkeleton.bones[b]
            let depthA = (projected[boneA.0].1 + projected[boneA.1].1) / 2
            let depthB = (projected[boneB.0].1 + projected[boneB.1].1) / 2
            return depthA > depthB  // further bones first
        }

        // Draw bones (depth-sorted)
        for idx in boneIndices {
            let bone = SMPLSkeleton.bones[idx]
            let p1 = projected[bone.0].0
            let p2 = projected[bone.1].0
            let avgDepth = (projected[bone.0].1 + projected[bone.1].1) / 2

            // Depth-based thickness and opacity
            let depthFactor = CGFloat((avgDepth / extent + 0.5).clamped(to: 0...1))
            let lineWidth: CGFloat = 3 + (1 - depthFactor) * 3  // 3-6 based on depth
            let opacity = 0.5 + (1 - depthFactor) * 0.5         // 0.5-1.0

            let c = SMPLSkeleton.boneColors[idx]
            let color = Color(red: Double(c.0), green: Double(c.1), blue: Double(c.2))

            var path = Path()
            path.move(to: p1)
            path.addLine(to: p2)
            context.stroke(path, with: .color(color.opacity(opacity)),
                           style: StrokeStyle(lineWidth: lineWidth, lineCap: .round))
        }

        // Draw joints (depth-sorted)
        let jointOrder = projected.indices.sorted { projected[$0].1 > projected[$1].1 }

        for i in jointOrder {
            let pt = projected[i].0
            let depth = projected[i].1
            let depthFactor = CGFloat((depth / extent + 0.5).clamped(to: 0...1))
            let radius: CGFloat = i == 0 ? 7 : (4 + (1 - depthFactor) * 3)
            let opacity = 0.6 + (1 - depthFactor) * 0.4

            let isLeft = SMPLSkeleton.jointNames[i].hasPrefix("L_")
            let isRight = SMPLSkeleton.jointNames[i].hasPrefix("R_")
            let fillColor: Color = isLeft ? .blue : (isRight ? .red : .white)

            let circle = Path(ellipseIn: CGRect(
                x: pt.x - radius, y: pt.y - radius,
                width: radius * 2, height: radius * 2
            ))
            context.fill(circle, with: .color(fillColor.opacity(opacity)))

            // Glow effect for closer joints
            if depthFactor < 0.4 {
                let glowRadius = radius + 3
                let glow = Path(ellipseIn: CGRect(
                    x: pt.x - glowRadius, y: pt.y - glowRadius,
                    width: glowRadius * 2, height: glowRadius * 2
                ))
                context.stroke(glow, with: .color(fillColor.opacity(0.3)), lineWidth: 1)
            }
        }
    }
}

// MARK: - Float Clamping

private extension Float {
    func clamped(to range: ClosedRange<Float>) -> Float {
        return Swift.min(Swift.max(self, range.lowerBound), range.upperBound)
    }
}
