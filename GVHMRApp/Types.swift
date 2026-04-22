// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Calvin Chan

import Foundation
import simd

// MARK: - Preprocessing Mode

/// Which preprocessing pipeline to use for 2D pose detection.
enum PreprocessingMode: String, CaseIterable, Identifiable {
    case visionViTPose = "Vision+ViTPose"
    case yoloViTPose = "YOLO+ViTPose"
    case yoloPose = "YOLO-Pose"

    var id: String { rawValue }

    var detail: String {
        switch self {
        case .visionViTPose: return "Apple Vision bbox + ViTPose-Small keypoints (~47 MB)"
        case .yoloViTPose: return "YOLO bbox + ViTPose keypoints — fast & accurate (~53 MB)"
        case .yoloPose: return "YOLO26n-Pose single-pass bbox + keypoints (~6 MB)"
        }
    }
}

// MARK: - Constants

/// GVHMR model variants available for inference.
/// All share the same I/O format, differing only in transformer size.
enum GVHMRModelChoice: String, CaseIterable, Identifiable {
    case small = "Small"
    case medium = "Medium"
    case original = "Original"

    var id: String { rawValue }

    /// CoreML model resource name in the app bundle.
    var modelName: String {
        switch self {
        case .small: return "GVHMRStudent"
        case .medium: return "GVHMRMedium"
        case .original: return "GVHMROriginal"
        }
    }

    /// Human-readable description.
    var detail: String {
        switch self {
        case .small: return "256d/6L/4H (~15 MB)"
        case .medium: return "384d/8L/6H (~35 MB)"
        case .original: return "512d/12L/8H (~60 MB)"
        }
    }
}

enum GVHMRConstants {
    static let windowSize = 16
    static let numJoints2D = 17    // COCO-17 keypoints
    static let numJointsSMPL = 22  // SMPL body joints
    static let imgseqDim = 1024
    static let outputDim = 151
    static let confidenceThreshold: Float = 0.5
    static let inferenceStride = 4  // run inference every N new frames

    // pred_x layout: [body_pose_r6d(126), betas(10), orient_c(6), orient_gv(6), transl_vel(3)]
    static let bodyPoseRange = 0..<126
    static let betasRange    = 126..<136
    static let orientCRange  = 136..<142
    static let orientGVRange = 142..<148
    static let translVelRange = 148..<151
}

// MARK: - Per-Frame Data

/// Data collected from one camera frame, ready for buffering.
struct FrameData {
    let keypoints: [SIMD3<Float>]    // 17 × (x_norm, y_norm, confidence)
    let cliffCam: SIMD3<Float>       // (cx-icx, cy-icy, bbox_size) / focal_length
    let camAngvel: [Float]           // 6-dim flattened 2×3 rotation
    let imageFeatures: [Float]       // 1024-dim from MobileNet proxy
    let boundingBox: CGRect          // person bbox in image coordinates
    let timestamp: TimeInterval
}

// MARK: - Inference Results

struct GVHMRResult {
    let joints3D: [SIMD3<Float>]     // 22 SMPL joints in camera frame
    let joints2D: [CGPoint]          // 22 joints projected to screen
    let predCam: SIMD3<Float>        // (s, tx, ty) weak-perspective camera
    let bodyPoseAA: [SIMD3<Float>]   // 21 joint rotations as axis-angle
    let globalOrient: SIMD3<Float>   // root orientation axis-angle
    let betas: [Float]               // 10 shape parameters
    let confidence: Float
    var meshVertices: [SIMD3<Float>]? = nil // 6890 SMPL mesh vertices (SceneKit coords: Y-up, Z-toward-viewer)
    var meshVerticesIncam: [SIMD3<Float>]? = nil // 6890 SMPL mesh vertices in camera coords (Y-down, Z-forward)
    var translFullCam: SIMD3<Float>? = nil  // full-perspective 3D translation in camera frame
}

// MARK: - SMPL Skeleton Definition

/// SMPL body model topology for 22 joints.
enum SMPLSkeleton {
    static let jointNames: [String] = [
        "Pelvis", "L_Hip", "R_Hip", "Spine1",
        "L_Knee", "R_Knee", "Spine2",
        "L_Ankle", "R_Ankle", "Spine3",
        "L_Foot", "R_Foot", "Neck",
        "L_Collar", "R_Collar", "Head",
        "L_Shoulder", "R_Shoulder",
        "L_Elbow", "R_Elbow",
        "L_Wrist", "R_Wrist"
    ]

    /// Parent index for each joint (-1 = root).
    static let parents: [Int] = [
        -1, 0, 0, 0,    // Pelvis, L_Hip, R_Hip, Spine1
         1, 2, 3,        // L_Knee, R_Knee, Spine2
         4, 5, 6,        // L_Ankle, R_Ankle, Spine3
         7, 8, 9,        // L_Foot, R_Foot, Neck
         9, 9, 12,       // L_Collar, R_Collar, Head
        13, 14,          // L_Shoulder, R_Shoulder
        16, 17,          // L_Elbow, R_Elbow
        18, 19           // L_Wrist, R_Wrist
    ]

    /// Bones to draw (parent, child pairs).
    static let bones: [(Int, Int)] = [
        (0, 1), (0, 2), (0, 3),       // pelvis to hips & spine
        (1, 4), (2, 5), (3, 6),       // hips to knees, spine1 to spine2
        (4, 7), (5, 8), (6, 9),       // knees to ankles, spine2 to spine3
        (7, 10), (8, 11), (9, 12),    // ankles to feet, spine3 to neck
        (9, 13), (9, 14), (12, 15),   // spine3 to collars, neck to head
        (13, 16), (14, 17),           // collars to shoulders
        (16, 18), (17, 19),           // shoulders to elbows
        (18, 20), (19, 21),           // elbows to wrists
    ]

    /// Default T-pose bone offsets (meters) for an average body.
    /// Each offset is from parent joint to this joint in the parent's local frame.
    static let defaultOffsets: [SIMD3<Float>] = [
        SIMD3<Float>(0, 0, 0),              // 0 Pelvis (root)
        SIMD3<Float>(0.083, -0.093, 0),      // 1 L_Hip
        SIMD3<Float>(-0.083, -0.093, 0),     // 2 R_Hip
        SIMD3<Float>(0, 0.143, 0),           // 3 Spine1
        SIMD3<Float>(0, -0.392, 0),          // 4 L_Knee
        SIMD3<Float>(0, -0.392, 0),          // 5 R_Knee
        SIMD3<Float>(0, 0.131, 0),           // 6 Spine2
        SIMD3<Float>(0, -0.429, 0),          // 7 L_Ankle
        SIMD3<Float>(0, -0.429, 0),          // 8 R_Ankle
        SIMD3<Float>(0, 0.123, 0),           // 9 Spine3
        SIMD3<Float>(0, -0.063, 0.096),      // 10 L_Foot
        SIMD3<Float>(0, -0.063, 0.096),      // 11 R_Foot
        SIMD3<Float>(0, 0.120, 0),           // 12 Neck
        SIMD3<Float>(0.01, 0, 0),            // 13 L_Collar
        SIMD3<Float>(-0.01, 0, 0),           // 14 R_Collar
        SIMD3<Float>(0, 0.110, 0),           // 15 Head
        SIMD3<Float>(0.155, 0, 0),           // 16 L_Shoulder
        SIMD3<Float>(-0.155, 0, 0),          // 17 R_Shoulder
        SIMD3<Float>(0.260, 0, 0),           // 18 L_Elbow
        SIMD3<Float>(-0.260, 0, 0),          // 19 R_Elbow
        SIMD3<Float>(0.250, 0, 0),           // 20 L_Wrist
        SIMD3<Float>(-0.250, 0, 0),          // 21 R_Wrist
    ]

    /// Colors for rendering: left side blue, right side red, spine green.
    static let boneColors: [(Float, Float, Float)] = [
        (0.2, 0.6, 1.0),  // 0-1  pelvis→L_Hip (blue)
        (1.0, 0.3, 0.3),  // 0-2  pelvis→R_Hip (red)
        (0.3, 0.9, 0.3),  // 0-3  pelvis→Spine1 (green)
        (0.2, 0.6, 1.0),  // 1-4  L_Hip→L_Knee
        (1.0, 0.3, 0.3),  // 2-5  R_Hip→R_Knee
        (0.3, 0.9, 0.3),  // 3-6  Spine1→Spine2
        (0.2, 0.6, 1.0),  // 4-7  L_Knee→L_Ankle
        (1.0, 0.3, 0.3),  // 5-8  R_Knee→R_Ankle
        (0.3, 0.9, 0.3),  // 6-9  Spine2→Spine3
        (0.2, 0.6, 1.0),  // 7-10 L_Ankle→L_Foot
        (1.0, 0.3, 0.3),  // 8-11 R_Ankle→R_Foot
        (0.3, 0.9, 0.3),  // 9-12 Spine3→Neck
        (0.2, 0.6, 1.0),  // 9-13 Spine3→L_Collar
        (1.0, 0.3, 0.3),  // 9-14 Spine3→R_Collar
        (0.9, 0.9, 0.3),  // 12-15 Neck→Head (yellow)
        (0.2, 0.6, 1.0),  // 13-16 L_Collar→L_Shoulder
        (1.0, 0.3, 0.3),  // 14-17 R_Collar→R_Shoulder
        (0.2, 0.6, 1.0),  // 16-18 L_Shoulder→L_Elbow
        (1.0, 0.3, 0.3),  // 17-19 R_Shoulder→R_Elbow
        (0.2, 0.6, 1.0),  // 18-20 L_Elbow→L_Wrist
        (1.0, 0.3, 0.3),  // 19-21 R_Elbow→R_Wrist
    ]
}

// MARK: - COCO-17 Keypoints

/// Mapping from Apple Vision body landmarks to COCO-17 ordering.
enum COCOKeypoints {
    static let names: [String] = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
}

// MARK: - Multi-Person Types

/// Distinct colors for up to 10 tracked persons (RGB, 0-1).
/// Matches the Python multi-HPE demo colors.
enum PersonColors {
    static let colors: [(Float, Float, Float)] = [
        (0.53, 0.81, 0.92),  // light blue
        (1.00, 0.60, 0.60),  // salmon
        (0.60, 0.90, 0.60),  // light green
        (1.00, 0.85, 0.40),  // gold
        (0.80, 0.60, 0.90),  // lavender
        (1.00, 0.70, 0.50),  // peach
        (0.40, 0.80, 0.80),  // teal
        (0.90, 0.50, 0.70),  // rose
        (0.70, 0.85, 0.55),  // lime
        (0.65, 0.65, 0.85),  // periwinkle
    ]

    static func color(for personIndex: Int) -> (Float, Float, Float) {
        colors[personIndex % colors.count]
    }
}

/// Per-person result for a single frame in multi-person mode.
struct PersonFrameResult {
    let personIndex: Int          // index within the frame's person list (for color)
    let trackID: Int              // stable tracking ID across frames
    let keypoints: [SIMD3<Float>] // 17 COCO keypoints in pixels
    let bbox: CGRect
    let bbxXYS: SIMD3<Float>
    let gvhmrResult: GVHMRResult
    let predX: [Float]
}

/// Multi-person frame result: all persons detected in a single frame.
struct MultiPersonFrameResult {
    let frameIndex: Int
    let persons: [PersonFrameResult]
}

// MARK: - SMPL Mesh Data

/// Loads SMPL triangle faces from the bundled binary file.
enum SMPLMeshData {
    /// Load face indices as a flat UInt32 array: [v0, v1, v2, v0, v1, v2, ...]
    static func loadFaces() -> [UInt32] {
        guard let url = Bundle.main.url(forResource: "smpl_faces", withExtension: "bin"),
              let data = try? Data(contentsOf: url),
              data.count >= 4
        else {
            print("WARNING: smpl_faces.bin not found in bundle")
            return []
        }

        let numFaces = data.withUnsafeBytes { $0.load(fromByteOffset: 0, as: UInt32.self) }
        let expectedBytes = 4 + Int(numFaces) * 3 * 4  // header + faces
        guard data.count >= expectedBytes else {
            print("WARNING: smpl_faces.bin too small (\(data.count) < \(expectedBytes))")
            return []
        }

        var faces = [UInt32]()
        faces.reserveCapacity(Int(numFaces) * 3)
        data.withUnsafeBytes { ptr in
            let facePtr = ptr.baseAddress!.advanced(by: 4).assumingMemoryBound(to: UInt32.self)
            for i in 0..<Int(numFaces) * 3 {
                faces.append(facePtr[i])
            }
        }
        return faces
    }
}
