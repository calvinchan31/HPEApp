import Foundation
import simd

// MARK: - Rotation Utilities

enum MathUtils {

    // MARK: Axis-Angle ↔ Rotation Matrix

    /// Convert axis-angle (3D vector whose norm = angle) to a 3×3 rotation matrix.
    /// Uses Rodrigues' formula: R = I + sin(θ)·K + (1−cos(θ))·K²
    static func axisAngleToMatrix(_ aa: SIMD3<Float>) -> simd_float3x3 {
        let angle = length(aa)
        if angle < 1e-6 {
            return matrix_identity_float3x3
        }
        let axis = aa / angle
        let K = skewSymmetric(axis)
        let s = sin(angle)
        let c = cos(angle)
        return matrix_identity_float3x3 + s * K + (1 - c) * (K * K)
    }

    /// Convert 3×3 rotation matrix to axis-angle.
    static func matrixToAxisAngle(_ R: simd_float3x3) -> SIMD3<Float> {
        let cosAngle = (R[0][0] + R[1][1] + R[2][2] - 1) / 2
        let angle = acos(min(max(cosAngle, -1), 1))
        if angle < 1e-6 {
            return SIMD3<Float>(0, 0, 0)
        }
        let axis = SIMD3<Float>(
            R[1][2] - R[2][1],
            R[2][0] - R[0][2],
            R[0][1] - R[1][0]
        ) / (2 * sin(angle))
        return normalize(axis) * angle
    }

    // MARK: 6D Rotation Representation

    /// Convert 6D rotation representation to 3×3 rotation matrix.
    /// Following Zhou et al. "On the Continuity of Rotation Representations in Neural Networks"
    /// PyTorch3D convention: the 6D values are the first two ROWS of the rotation matrix.
    static func rotation6DToMatrix(_ r6d: [Float]) -> simd_float3x3 {
        assert(r6d.count == 6)
        let a1 = SIMD3<Float>(r6d[0], r6d[1], r6d[2])
        let a2 = SIMD3<Float>(r6d[3], r6d[4], r6d[5])

        let b1 = safeNormalize(a1)
        let b2 = safeNormalize(a2 - dot(b1, a2) * b1)
        let b3 = cross(b1, b2)

        // b1, b2, b3 are ROWS of the rotation matrix (PyTorch3D convention).
        // simd_float3x3(b1, b2, b3) would place them as columns, so we transpose.
        return simd_float3x3(b1, b2, b3).transpose
    }

    /// Convert 3×3 rotation matrix to 6D rotation representation.
    /// PyTorch3D convention: returns the first two ROWS flattened.
    static func matrixToRotation6D(_ R: simd_float3x3) -> [Float] {
        // First two rows. In simd column-major: row i, col j = R[j][i]
        return [R[0][0], R[1][0], R[2][0], R[0][1], R[1][1], R[2][1]]
    }

    // MARK: Helpers

    static func skewSymmetric(_ v: SIMD3<Float>) -> simd_float3x3 {
        return simd_float3x3(
            SIMD3<Float>(0, v.z, -v.y),
            SIMD3<Float>(-v.z, 0, v.x),
            SIMD3<Float>(v.y, -v.x, 0)
        )
    }

    static func safeNormalize(_ v: SIMD3<Float>) -> SIMD3<Float> {
        let n = length(v)
        return n > 1e-8 ? v / n : SIMD3<Float>(1, 0, 0)
    }

    // MARK: 4×4 Transform Matrix

    /// Build a 4×4 transform from rotation and translation.
    static func makeTransform(rotation R: simd_float3x3, translation t: SIMD3<Float>) -> simd_float4x4 {
        var m = simd_float4x4(1)  // identity
        // Set rotation (upper-left 3×3)
        m[0] = SIMD4<Float>(R[0], 0)
        m[1] = SIMD4<Float>(R[1], 0)
        m[2] = SIMD4<Float>(R[2], 0)
        // Set translation (last column)
        m[3] = SIMD4<Float>(t, 1)
        return m
    }

    /// Extract translation from 4×4 transform matrix.
    static func getTranslation(_ m: simd_float4x4) -> SIMD3<Float> {
        return SIMD3<Float>(m[3].x, m[3].y, m[3].z)
    }

    /// Multiply point by 4×4 matrix (homogeneous).
    static func transformPoint(_ m: simd_float4x4, _ p: SIMD3<Float>) -> SIMD3<Float> {
        let p4 = m * SIMD4<Float>(p, 1)
        return SIMD3<Float>(p4.x, p4.y, p4.z)
    }

    // MARK: Weak-Perspective Projection

    /// Project 3D joint to 2D using weak-perspective camera (s, tx, ty).
    /// Returns normalized coordinates centered at origin.
    static func weakPerspectiveProject(
        joint3D: SIMD3<Float>,
        s: Float, tx: Float, ty: Float
    ) -> CGPoint {
        let x = s * joint3D.x + tx
        let y = s * joint3D.y + ty
        return CGPoint(x: CGFloat(x), y: CGFloat(y))
    }
}
