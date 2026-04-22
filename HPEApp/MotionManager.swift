import CoreMotion
import Foundation
import simd

/// Manages device motion data from the gyroscope/accelerometer.
/// Provides camera angular velocity as a 6-dim vector (flattened 2×3 rotation matrix).
class MotionManager: ObservableObject {
    private let motionManager = CMMotionManager()
    private let queue = OperationQueue()

    /// Previous rotation matrix for computing inter-frame rotation.
    private var prevRotation: simd_float3x3?

    /// Latest camera angular velocity (normalized 6D rotation representation).
    /// Format: first two columns of inter-frame rotation, then normalized
    /// using the "manual" stats from GVHMR training.
    @Published var angularVelocity: [Float] = [0, 0, 0, 0, 0, 0]  // normalized identity

    // cam_angvel normalization stats (from stats_compose.cam_angvel["manual"])
    private let angvelMean: [Float] = [1, 0, 0, 0, 1, 0]
    private let angvelStd: [Float]  = [0.001, 0.1, 0.1, 0.1, 0.001, 0.1]

    init() {
        queue.name = "com.gvhmr.motion"
        queue.maxConcurrentOperationCount = 1
    }

    func start() {
        guard motionManager.isDeviceMotionAvailable else {
            print("WARNING: Device motion not available")
            return
        }

        motionManager.deviceMotionUpdateInterval = 1.0 / 30.0
        motionManager.startDeviceMotionUpdates(using: .xArbitraryZVertical, to: queue) { [weak self] motion, _ in
            guard let self = self, let motion = motion else { return }
            self.processMotion(motion)
        }
    }

    func stop() {
        motionManager.stopDeviceMotionUpdates()
        prevRotation = nil
    }

    private func processMotion(_ motion: CMDeviceMotion) {
        // Get the current rotation as a 3×3 matrix
        let rm = motion.attitude.rotationMatrix
        let currentRotation = simd_float3x3(
            SIMD3<Float>(Float(rm.m11), Float(rm.m21), Float(rm.m31)),
            SIMD3<Float>(Float(rm.m12), Float(rm.m22), Float(rm.m32)),
            SIMD3<Float>(Float(rm.m13), Float(rm.m23), Float(rm.m33))
        )

        // Compute relative rotation: R_rel = R_current * R_prev^T
        if let prev = prevRotation {
            let relRotation = currentRotation * prev.transpose
            // Extract first 2 columns of rotation matrix to match
            // PyTorch3D's matrix_to_rotation_6d format:
            // [col0_row0, col0_row1, col0_row2, col1_row0, col1_row1, col1_row2]
            // In simd_float3x3: [colIdx][rowIdx]
            let raw: [Float] = [
                relRotation[0][0], relRotation[0][1], relRotation[0][2],
                relRotation[1][0], relRotation[1][1], relRotation[1][2]
            ]
            // Normalize: (x - mean) / std
            var normalized = [Float](repeating: 0, count: 6)
            for i in 0..<6 {
                normalized[i] = (raw[i] - angvelMean[i]) / angvelStd[i]
            }
            DispatchQueue.main.async {
                self.angularVelocity = normalized
            }
        }

        prevRotation = currentRotation
    }

    /// Get current angular velocity (thread-safe copy).
    func getCurrentAngvel() -> [Float] {
        return angularVelocity
    }
}
