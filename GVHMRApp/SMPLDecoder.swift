// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Calvin Chan
// 
// SMPL body model forward kinematics and projection.
// SMPL licensing: See ACKNOWLEDGMENTS.md

import Foundation
import simd

/// Decodes the GVHMR model output (151-dim normalized vector) into SMPL body parameters
/// and computes 3D joint positions via forward kinematics.
class SMPLDecoder {

    let mean: [Float]
    let std: [Float]
    let predCamMean: SIMD3<Float>
    let predCamStd: SIMD3<Float>
    let offsets: [SIMD3<Float>]
    let parents: [Int]

    init() {
        // Load statistics from bundle
        let stats = SMPLDecoder.loadStats()
        self.mean = stats.mean
        self.std = stats.std
        self.predCamMean = SIMD3<Float>(stats.predCamMean[0], stats.predCamMean[1], stats.predCamMean[2])
        self.predCamStd = SIMD3<Float>(stats.predCamStd[0], stats.predCamStd[1], stats.predCamStd[2])
        self.offsets = SMPLSkeleton.defaultOffsets
        self.parents = SMPLSkeleton.parents
    }

    // MARK: - Decode

    /// Decode a single frame's pred_x (151-dim normalized) into 3D joints.
    /// - Parameters:
    ///   - predX: normalized 151-dim output from GVHMR
    ///   - predCam: 3-dim camera params (s, tx, ty) — already denormalized by model
    /// - Returns: GVHMRResult with 3D and 2D joints
    func decode(predX: [Float], predCam: SIMD3<Float>, imageSize: CGSize) -> GVHMRResult {
        // 1. Denormalize pred_x: x = x_norm * std + mean
        //    Note: pred_cam is NOT denormalized here — the model already applies
        //    pred_cam = raw * pred_cam_std + pred_cam_mean internally.
        var x = [Float](repeating: 0, count: GVHMRConstants.outputDim)
        for i in 0..<GVHMRConstants.outputDim {
            x[i] = predX[i] * std[i] + mean[i]
        }

        // 2. Extract components
        let bodyPoseR6D = Array(x[GVHMRConstants.bodyPoseRange])   // 126 = 21 * 6
        let betas = Array(x[GVHMRConstants.betasRange])            // 10
        let orientCR6D = Array(x[GVHMRConstants.orientCRange])     // 6
        // orientGVR6D and translVel are available but not needed for rendering

        // 3. Convert 6D rotations to rotation matrices
        let globalOrientMat = MathUtils.rotation6DToMatrix(orientCR6D)
        let globalOrientAA = MathUtils.matrixToAxisAngle(globalOrientMat)

        var jointRotations = [simd_float3x3]()
        jointRotations.reserveCapacity(21)
        var bodyPoseAA = [SIMD3<Float>]()
        bodyPoseAA.reserveCapacity(21)

        for j in 0..<21 {
            let r6d = Array(bodyPoseR6D[j*6..<(j+1)*6])
            let rotMat = MathUtils.rotation6DToMatrix(r6d)
            jointRotations.append(rotMat)
            bodyPoseAA.append(MathUtils.matrixToAxisAngle(rotMat))
        }

        // 4. Forward kinematics
        let joints3D = forwardKinematics(
            globalOrient: globalOrientMat,
            bodyPose: jointRotations
        )

        // 5. Use pred_cam directly — the model already denormalizes
        //    pred_cam = raw * std + mean inside its forward() method
        let s = max(predCam.x, 0.25)  // match model's clamp_min
        let tx = predCam.y
        let ty = predCam.z
        var joints2D = [CGPoint]()
        joints2D.reserveCapacity(22)
        for joint in joints3D {
            let pt = MathUtils.weakPerspectiveProject(joint3D: joint, s: s, tx: tx, ty: ty)
            // Map from normalized coords to screen coords
            // The model outputs coordinates roughly in [-1, 1] range
            let screenX = (CGFloat(pt.x) + 1) / 2 * imageSize.width
            let screenY = (CGFloat(pt.y) + 1) / 2 * imageSize.height
            joints2D.append(CGPoint(x: screenX, y: screenY))
        }

        return GVHMRResult(
            joints3D: joints3D,
            joints2D: joints2D,
            predCam: predCam,
            bodyPoseAA: bodyPoseAA,
            globalOrient: globalOrientAA,
            betas: betas,
            confidence: 1.0
        )
    }

    // MARK: - Forward Kinematics

    /// Compute 22 joint positions using forward kinematics.
    /// - Parameters:
    ///   - globalOrient: root rotation matrix (3×3)
    ///   - bodyPose: 21 local joint rotation matrices
    /// - Returns: 22 joint positions in camera coordinates
    func forwardKinematics(
        globalOrient: simd_float3x3,
        bodyPose: [simd_float3x3]
    ) -> [SIMD3<Float>] {
        // Build local transforms (rotation + bone offset translation)
        var localTransforms = [simd_float4x4]()
        localTransforms.reserveCapacity(22)

        // Joint 0: root with global orientation
        localTransforms.append(MathUtils.makeTransform(rotation: globalOrient, translation: offsets[0]))

        // Joints 1-21: local pose rotations
        for j in 1..<22 {
            let rotation = bodyPose[j - 1]
            localTransforms.append(MathUtils.makeTransform(rotation: rotation, translation: offsets[j]))
        }

        // Forward kinematics: compute world transforms by chaining parent transforms
        var worldTransforms = [simd_float4x4](repeating: simd_float4x4(1), count: 22)
        worldTransforms[0] = localTransforms[0]

        for j in 1..<22 {
            let parentIdx = parents[j]
            worldTransforms[j] = worldTransforms[parentIdx] * localTransforms[j]
        }

        // Extract joint positions
        return worldTransforms.map { MathUtils.getTranslation($0) }
    }

    // MARK: - Full Camera Translation

    /// Convert pred_cam (s, tx, ty) → full 3D camera translation.
    /// Matches Python: compute_transl_full_cam(pred_cam, bbx_xys, K_fullimg)
    func computeTranslFullCam(predCam: SIMD3<Float>, bbxXYS: SIMD3<Float>, focalLength: Float, imageSize: CGSize) -> SIMD3<Float> {
        let s = predCam.x
        let tx = predCam.y
        let ty = predCam.z

        let icx = Float(imageSize.width) / 2
        let icy = Float(imageSize.height) / 2
        let sb = s * bbxXYS.z

        let cx = 2 * (bbxXYS.x - icx) / (sb + 1e-9)
        let cy = 2 * (bbxXYS.y - icy) / (sb + 1e-9)
        let tz = 2 * focalLength / (sb + 1e-9)

        return SIMD3<Float>(tx + cx, ty + cy, tz)
    }

    // MARK: - Statistics Loading

    struct Stats {
        let mean: [Float]
        let std: [Float]
        let predCamMean: [Float]
        let predCamStd: [Float]
    }

    static func loadStats() -> Stats {
        guard let url = Bundle.main.url(forResource: "gvhmr_stats", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let mean = json["mean"] as? [Double],
              let std = json["std"] as? [Double],
              let pcm = json["pred_cam_mean"] as? [Double],
              let pcs = json["pred_cam_std"] as? [Double]
        else {
            print("WARNING: Could not load gvhmr_stats.json, using defaults")
            return Stats(
                mean: [Float](repeating: 0, count: 151),
                std: [Float](repeating: 1, count: 151),
                predCamMean: [1.0606, -0.0027, 0.2702],
                predCamStd: [0.1784, 0.0956, 0.0764]
            )
        }
        return Stats(
            mean: mean.map { Float($0) },
            std: std.map { Float($0) },
            predCamMean: pcm.map { Float($0) },
            predCamStd: pcs.map { Float($0) }
        )
    }
}
