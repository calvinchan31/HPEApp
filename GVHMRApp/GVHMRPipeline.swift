// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Calvin Chan

import AVFoundation
import Combine
import simd
import UIKit

enum LiveCaptionMode: String, CaseIterable {
    case captioner = "Captioner"
    case pose = "PoseCaption"
}

/// Orchestrates the full GVHMR pipeline:
///   Camera frame → 2D pose detection → feature extraction → GVHMR inference → 3D skeleton
class GVHMRPipeline: ObservableObject {

    // MARK: - Published State

    @Published var currentResult: GVHMRResult?
    @Published var inputKeypoints: [SIMD3<Float>] = []
    @Published var inputBBox: CGRect = .zero
    @Published var fps: Double = 0
    @Published var status: String = "Initializing..."
    @Published var isRunning = false
    @Published var debugInfo: String = ""
    @Published var smplFaces: [UInt32] = []
    @Published var preprocessingMode: PreprocessingMode = .yoloViTPose
    @Published var isMultiPerson: Bool = false
    /// Multi-person 2D inputs for overlay (live mode: detection + tracking only, no per-person GVHMR).
    @Published var multiPersonInputs: [(keypoints: [SIMD3<Float>], bbox: CGRect, trackID: Int)] = []
    /// Live multi-person SMPL outputs (one per active tracked person).
    @Published var liveMultiPersonResults: [PersonFrameResult] = []
    /// Enable/disable semantic captions over live output.
    @Published var captionEnabled: Bool = true
    @Published var captionText: String = "Semantic stream idle"
    @Published var captionConfidence: Float = 0
    @Published var captionModelStatus: String = "No Caption model"
    @Published var captionMode: LiveCaptionMode = .pose
    @Published var selectedTrackID: Int? = nil
    @Published var visiblePersonCount: Int = 0
    @Published var personCaptionFrames: [Int: CaptionSemanticFrame] = [:]

    let metrics = PerformanceMetrics()
    private var metricsCancellable: AnyCancellable?

    // MARK: - Components

    let cameraManager = CameraManager()
    let motionManager = MotionManager()
    private let poseDetector = PoseDetector()
    private let vitPose = VitPoseProcessor()
    private let yoloPose = YOLOPoseProcessor()
    private let inference = GVHMRInference()
    private let smplDecoder = SMPLDecoder()
    private let captionFusion = CaptionFusionEngine()
    private let captionTexter = CoreMLCaptioner()
    private let frameBuffer = FrameBuffer()
    private let personTracker = PersonTracker()
    private var personBuffers: [Int: FrameBuffer] = [:]
    private var latestPersonResults: [Int: PersonFrameResult] = [:]
    private var frameSerial: Int = 0
    private var captionFrameIndex: Int = 0
    private var lastVisualCaption: String?
    private var lastVisualCaptionFrame: Int = -1
    private let captionVisualStride = 8
    private var perPersonCaptionFusion: [Int: CaptionFusionEngine] = [:]
    private var perPersonCaptionFrameIndex: [Int: Int] = [:]
    private var perPersonVisualCaption: [Int: String] = [:]
    private var perPersonVisualCaptionFrame: [Int: Int] = [:]

    /// Available model variants for the UI.
    var availableModels: [GVHMRModelChoice] {
        return inference.availableModels
    }

    /// Switch which GVHMR model is used for inference.
    func selectModel(_ choice: GVHMRModelChoice) {
        inference.selectModel(choice)
        metrics.resetLive()
    }

    /// Switch preprocessing mode.
    func selectPreprocessing(_ mode: PreprocessingMode) {
        DispatchQueue.main.async { [weak self] in
            self?.preprocessingMode = mode
        }
    }

    func selectPerson(trackID: Int?) {
        DispatchQueue.main.async { [weak self] in
            self?.selectedTrackID = trackID
        }
    }

    func toggleLiveCaptionMode() {
        if !captionTexter.isReady {
            captionMode = .pose
            if captionEnabled {
                captionText = "PoseCaption active (visual model unavailable)"
            }
            return
        }

        captionMode = (captionMode == .captioner) ? .pose : .captioner
        lastVisualCaption = nil
        lastVisualCaptionFrame = -1
        perPersonVisualCaption.removeAll()
        perPersonVisualCaptionFrame.removeAll()

        if captionEnabled {
            captionText = (captionMode == .captioner) ? "Captioner active" : "PoseCaption active"
        }
    }

    // MARK: - Timing

    private var frameCount = 0
    private var lastFPSTime = Date()
    private let processingQueue = DispatchQueue(label: "com.gvhmr.pipeline", qos: .userInitiated)

    // MARK: - Lifecycle

    func setup() {
        // Forward metrics changes so SwiftUI observes nested ObservableObject
        metricsCancellable = metrics.objectWillChange.sink { [weak self] _ in
            self?.objectWillChange.send()
        }
        DispatchQueue.main.async { [weak self] in
            self?.status = "Loading models..."
        }
        inference.loadModels()
        vitPose.loadModel()
        yoloPose.loadModel()
        captionTexter.loadModel()

        let faces = SMPLMeshData.loadFaces()

        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.smplFaces = faces
            if self.inference.isReady {
                let smplStatus = self.inference.smplReady ? " + SMPL mesh" : ""
                self.status = "Models loaded\(smplStatus). Tap Start."
            } else {
                self.status = "Models not found — see README for setup"
            }
            self.debugInfo = self.inference.diagnosticInfo
            self.captionModelStatus = self.captionTexter.isReady
                ? "CoreML Caption loaded + PoseCaption"
                : "PoseCaption active (CoreML visual model missing)"
            self.captionMode = self.captionTexter.isReady ? .captioner : .pose
        }

        // Wire up camera frame callback
        cameraManager.onFrame = { [weak self] pixelBuffer, timestamp in
            self?.processingQueue.async {
                self?.processFrame(pixelBuffer: pixelBuffer, timestamp: timestamp)
            }
        }
    }

    func start() {
        frameBuffer.reset()
        personTracker.reset()
        personBuffers.removeAll()
        latestPersonResults.removeAll()
        captionFusion.reset()
        captionFrameIndex = 0
        lastVisualCaption = nil
        lastVisualCaptionFrame = -1
        perPersonCaptionFusion.removeAll()
        perPersonCaptionFrameIndex.removeAll()
        perPersonVisualCaption.removeAll()
        perPersonVisualCaptionFrame.removeAll()
        frameSerial = 0
        cameraManager.start()
        motionManager.start()
        DispatchQueue.main.async { [weak self] in
            self?.isRunning = true
            self?.status = "Running"
            self?.captionText = "Semantic stream warming up"
            self?.captionConfidence = 0
            self?.personCaptionFrames = [:]
            self?.selectedTrackID = nil
            self?.visiblePersonCount = 0
        }
    }

    /// Clear transient runtime buffers to reduce current memory footprint.
    /// Safe while running because all processing work is serialized on processingQueue.
    func clearRAM() {
        processingQueue.async { [weak self] in
            guard let self else { return }

            self.frameBuffer.reset()
            self.personTracker.reset()
            self.personBuffers.removeAll()
            self.latestPersonResults.removeAll()

            self.captionFusion.reset()
            self.captionFrameIndex = 0
            self.lastVisualCaption = nil
            self.lastVisualCaptionFrame = -1

            self.perPersonCaptionFusion.removeAll()
            self.perPersonCaptionFrameIndex.removeAll()
            self.perPersonVisualCaption.removeAll()
            self.perPersonVisualCaptionFrame.removeAll()
            self.personCaptionFramesCache.removeAll()

            self.frameCount = 0
            self.lastFPSTime = Date()
            self.metrics.resetLive()
            URLCache.shared.removeAllCachedResponses()

            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.currentResult = nil
                self.inputKeypoints = []
                self.inputBBox = .zero
                self.multiPersonInputs = []
                self.liveMultiPersonResults = []
                self.personCaptionFrames = [:]
                self.selectedTrackID = nil
                self.visiblePersonCount = 0
                self.captionConfidence = 0
                self.captionText = self.captionEnabled
                    ? "Semantic stream warming up"
                    : "Semantic stream disabled"
                self.status = self.isRunning ? "Running (RAM cleared)" : "Stopped (RAM cleared)"
                self.metrics.updateMemory()
            }
        }
    }

    func stop() {
        cameraManager.stop()
        motionManager.stop()
        personBuffers.removeAll()
        latestPersonResults.removeAll()
        captionFusion.reset()
        captionFrameIndex = 0
        lastVisualCaption = nil
        lastVisualCaptionFrame = -1
        perPersonCaptionFusion.removeAll()
        perPersonCaptionFrameIndex.removeAll()
        perPersonVisualCaption.removeAll()
        perPersonVisualCaptionFrame.removeAll()
        DispatchQueue.main.async { [weak self] in
            self?.isRunning = false
            self?.status = "Stopped"
            self?.liveMultiPersonResults = []
            self?.captionText = "Semantic stream idle"
            self?.captionConfidence = 0
            self?.personCaptionFrames = [:]
            self?.selectedTrackID = nil
            self?.visiblePersonCount = 0
        }
    }

    // MARK: - Frame Processing

    private func processFrame(pixelBuffer: CVPixelBuffer, timestamp: CMTime) {
        frameSerial += 1
        let imageSize = cameraManager.imageSize
        let fl = cameraManager.focalLength
        let ppx = Float(cameraManager.principalPoint.x)
        let ppy = Float(cameraManager.principalPoint.y)

        // 1. Detect 2D body pose — using selected preprocessing mode
        let detectStart = CFAbsoluteTimeGetCurrent()

        let bbox: CGRect
        let keypoints: [SIMD3<Float>]

        if preprocessingMode == .yoloPose && yoloPose.isReady {
            // YOLO-Pose: single pass for bbox + keypoints
            guard let yoloResult = yoloPose.detect(pixelBuffer: pixelBuffer, imageSize: imageSize) else {
                return
            }
            metrics.recordPoseDetect(CFAbsoluteTimeGetCurrent() - detectStart)
            bbox = yoloResult.bbox
            keypoints = yoloResult.keypoints
        } else if preprocessingMode == .yoloViTPose && yoloPose.isReady {
            // Hybrid: YOLO for fast bbox + ViTPose for quality keypoints
            guard let yoloResult = yoloPose.detect(pixelBuffer: pixelBuffer, imageSize: imageSize) else {
                return
            }
            metrics.recordPoseDetect(CFAbsoluteTimeGetCurrent() - detectStart)
            bbox = yoloResult.bbox

            if vitPose.isReady {
                let vitStart = CFAbsoluteTimeGetCurrent()
                let bbxXYS_vit = poseDetector.computeBBXXYS(bbox: bbox)
                if let vitKP = vitPose.extractKeypoints(
                       pixelBuffer: pixelBuffer, bbxXYS: bbxXYS_vit, imageSize: imageSize) {
                    keypoints = vitKP
                    metrics.recordVitpose(CFAbsoluteTimeGetCurrent() - vitStart)
                } else {
                    keypoints = yoloResult.keypoints
                }
            } else {
                keypoints = yoloResult.keypoints
            }
        } else {
            // Vision + ViTPose: two-step pipeline
            guard let detection = poseDetector.detect(pixelBuffer: pixelBuffer, imageSize: imageSize) else {
                return
            }
            metrics.recordPoseDetect(CFAbsoluteTimeGetCurrent() - detectStart)
            bbox = detection.bbox

            // Use ViTPose for keypoints if available (matches GVHMR training data)
            if vitPose.isReady {
                let vitStart = CFAbsoluteTimeGetCurrent()
                let bbxXYS_vit = poseDetector.computeBBXXYS(bbox: bbox)
                if let vitKP = vitPose.extractKeypoints(
                       pixelBuffer: pixelBuffer, bbxXYS: bbxXYS_vit, imageSize: imageSize) {
                    keypoints = vitKP
                    metrics.recordVitpose(CFAbsoluteTimeGetCurrent() - vitStart)
                } else {
                    keypoints = detection.keypoints
                }
            } else {
                keypoints = detection.keypoints
            }
        }

        // Update input visualization on main thread
        DispatchQueue.main.async { [weak self] in
            self?.inputKeypoints = keypoints
            self?.inputBBox = bbox
        }

        // Multi-person: detect all persons and track them for colored overlay
        if isMultiPerson && yoloPose.isReady {
            let allDetections = yoloPose.detectAll(pixelBuffer: pixelBuffer, imageSize: imageSize, maxPersons: 5)
            let tracked = personTracker.update(detections: allDetections)

            // Keep only currently active tracks in caches
            let activeIDs = Set(tracked.map { $0.trackID })
            personBuffers = personBuffers.filter { activeIDs.contains($0.key) }
            latestPersonResults = latestPersonResults.filter { activeIDs.contains($0.key) }
            perPersonCaptionFusion = perPersonCaptionFusion.filter { activeIDs.contains($0.key) }
            perPersonCaptionFrameIndex = perPersonCaptionFrameIndex.filter { activeIDs.contains($0.key) }
            perPersonVisualCaption = perPersonVisualCaption.filter { activeIDs.contains($0.key) }
            perPersonVisualCaptionFrame = perPersonVisualCaptionFrame.filter { activeIDs.contains($0.key) }
            personCaptionFramesCache = personCaptionFramesCache.filter { activeIDs.contains($0.key) }

            // Build/update per-person temporal buffers and run per-person GVHMR
            for person in tracked {
                let bbxXYS = poseDetector.computeBBXXYS(bbox: person.bbox)
                let normalizedKP = poseDetector.normalizeKeypoints(person.keypoints, bbxXYS: bbxXYS)
                let cliffCam = SIMD3<Float>(
                    (bbxXYS.x - ppx) / fl,
                    (bbxXYS.y - ppy) / fl,
                    bbxXYS.z / fl
                )
                let camAngvel = motionManager.getCurrentAngvel()
                let imageFeatures = [Float](repeating: 0, count: GVHMRConstants.imgseqDim)
                let frameData = FrameData(
                    keypoints: normalizedKP,
                    cliffCam: cliffCam,
                    camAngvel: camAngvel,
                    imageFeatures: imageFeatures,
                    boundingBox: person.bbox,
                    timestamp: timestamp.seconds
                )

                if personBuffers[person.trackID] == nil {
                    personBuffers[person.trackID] = FrameBuffer()
                }
                personBuffers[person.trackID]?.append(frameData)

                if let buffer = personBuffers[person.trackID],
                   buffer.shouldRunInference(),
                   inference.isReady,
                   let personResult = runInferenceForPerson(
                        buffer: buffer,
                        keypoints: person.keypoints,
                        bbox: person.bbox,
                        bbxXYS: bbxXYS,
                        trackID: person.trackID,
                        imageSize: imageSize,
                        focalLength: fl
                   ) {
                    latestPersonResults[person.trackID] = personResult
                }
            }

            let sortedLive = latestPersonResults.values.sorted { $0.trackID < $1.trackID }
            if !sortedLive.isEmpty {
                for person in sortedLive {
                    updateLiveCaptionForPerson(
                        personResult: person,
                        personCount: sortedLive.count,
                        timestampSec: timestamp.seconds,
                        pixelBuffer: pixelBuffer,
                        imageSize: imageSize
                    )
                }
            }
            DispatchQueue.main.async { [weak self] in
                guard let self else { return }
                self.multiPersonInputs = tracked.map {
                    (keypoints: $0.keypoints, bbox: $0.bbox, trackID: $0.trackID)
                }
                self.liveMultiPersonResults = sortedLive
                self.visiblePersonCount = sortedLive.count
                self.personCaptionFrames = self.personCaptionFramesCache

                if let selected = self.selectedTrackID,
                   !sortedLive.contains(where: { $0.trackID == selected }) {
                    self.selectedTrackID = sortedLive.first?.trackID
                } else if self.selectedTrackID == nil {
                    self.selectedTrackID = sortedLive.first?.trackID
                }

                if let selected = self.selectedTrackID,
                   let sem = self.personCaptionFramesCache[selected] {
                    self.captionText = sem.caption
                    self.captionConfidence = sem.confidence
                }

                // Keep backward-compatible single person bindings in sync with primary person.
                if let primary = sortedLive.first {
                    self.currentResult = primary.gvhmrResult
                    self.inputKeypoints = primary.keypoints
                    self.inputBBox = primary.bbox
                }
            }

            // In multi-person mode we skip the single global frameBuffer path.
            updateFPS()
            if frameCount % 30 == 0 { metrics.updateMemory() }
            return
        } else {
            DispatchQueue.main.async { [weak self] in
                self?.multiPersonInputs = []
                self?.liveMultiPersonResults = []
                self?.personCaptionFrames = [:]
                self?.selectedTrackID = nil
                self?.visiblePersonCount = 0
                self?.captionText = "No person selected"
            }
        }

        // 2. Normalize keypoints relative to bbox
        let bbxXYS = poseDetector.computeBBXXYS(bbox: bbox)
        let normalizedKP = poseDetector.normalizeKeypoints(keypoints, bbxXYS: bbxXYS)

        // 3. Compute CLIFF camera parameters

        let cliffCam = SIMD3<Float>(
            (bbxXYS.x - ppx) / fl,
            (bbxXYS.y - ppy) / fl,
            bbxXYS.z / fl
        )

        // 4. Get camera angular velocity from gyroscope
        let camAngvel = motionManager.getCurrentAngvel()

        // 5. Zero image features — ablation testing showed that the distilled
        //    MobileNet proxy features hurt quality more than zero features.
        let imageFeatures = [Float](repeating: 0, count: GVHMRConstants.imgseqDim)

        // 6. Buffer this frame
        let frameData = FrameData(
            keypoints: normalizedKP,
            cliffCam: cliffCam,
            camAngvel: camAngvel,
            imageFeatures: imageFeatures,
            boundingBox: bbox,
            timestamp: timestamp.seconds
        )
        frameBuffer.append(frameData)

        // 7. Update buffering status
        let bufCount = frameBuffer.frames.count
        let bufCap = frameBuffer.capacity
        let modelsOK = inference.isReady
        DispatchQueue.main.async { [weak self] in
            if !modelsOK {
                self?.debugInfo = "Models not loaded"
            } else if bufCount < bufCap {
                self?.debugInfo = "Buffering \(bufCount)/\(bufCap)"
            }
        }

        // 8. Run GVHMR inference when ready
        if frameBuffer.shouldRunInference() && inference.isReady {
            runInference(imageSize: imageSize, timestampSec: timestamp.seconds)
        }

        // Update FPS counter
        updateFPS()

        // Update memory periodically
        if frameCount % 30 == 0 {
            metrics.updateMemory()
        }
    }

    private func runInference(imageSize: CGSize, timestampSec: Double) {
        let pipelineStart = CFAbsoluteTimeGetCurrent()

        let obs = frameBuffer.packKeypoints()
        let cliff = frameBuffer.packCliffCam()
        let angvel = frameBuffer.packCamAngvel()
        let imgseq = frameBuffer.packImageFeatures()

        let gvhmrStart = CFAbsoluteTimeGetCurrent()
        guard let output = inference.runGVHMR(
            obs: obs, cliffCam: cliff, camAngvel: angvel, imgseq: imgseq
        ) else {
            DispatchQueue.main.async { [weak self] in
                self?.debugInfo = "GVHMR inference returned nil"
            }
            return
        }
        metrics.recordGVHMR(CFAbsoluteTimeGetCurrent() - gvhmrStart)

        // Decode the last frame's prediction (most recent)
        let lastIdx = GVHMRConstants.windowSize - 1
        let predX = output.predX[lastIdx]
        let predCam = output.predCam[lastIdx]

        let result = smplDecoder.decode(predX: predX, predCam: predCam, imageSize: imageSize)

        // Run SMPL mesh if available
        var finalResult = result
        if inference.smplReady {
            let bodyPoseFlat = result.bodyPoseAA.flatMap { [$0.x, $0.y, $0.z] }
            let orientFlat = [result.globalOrient.x, result.globalOrient.y, result.globalOrient.z]
            let smplStart = CFAbsoluteTimeGetCurrent()
            if let incamVerts = inference.runSMPLIncam(
                bodyPoseAA: bodyPoseFlat,
                globalOrientAA: orientFlat,
                betas: result.betas
            ) {
                metrics.recordSMPL(CFAbsoluteTimeGetCurrent() - smplStart)
                finalResult.meshVerticesIncam = incamVerts
                finalResult.meshVertices = incamVerts.map { SIMD3<Float>($0.x, -$0.y, -$0.z) }
            }
        }

        metrics.recordPipeline(CFAbsoluteTimeGetCurrent() - pipelineStart)

        if captionEnabled {
            updateLiveCaption(
                result: finalResult,
                personCount: 1,
                timestampSec: timestampSec,
                pixelBuffer: nil
            )
        }

        DispatchQueue.main.async { [weak self] in
            self?.currentResult = finalResult
            let meshInfo = finalResult.meshVertices != nil ? " + mesh" : ""
            self?.debugInfo = "\(finalResult.joints3D.count) joints\(meshInfo)"
        }
    }

    private func runInferenceForPerson(
        buffer: FrameBuffer,
        keypoints: [SIMD3<Float>],
        bbox: CGRect,
        bbxXYS: SIMD3<Float>,
        trackID: Int,
        imageSize: CGSize,
        focalLength: Float
    ) -> PersonFrameResult? {
        let pipelineStart = CFAbsoluteTimeGetCurrent()

        let obs = buffer.packKeypoints()
        let cliff = buffer.packCliffCam()
        let angvel = buffer.packCamAngvel()
        let imgseq = buffer.packImageFeatures()

        let gvhmrStart = CFAbsoluteTimeGetCurrent()
        guard let output = inference.runGVHMR(obs: obs, cliffCam: cliff, camAngvel: angvel, imgseq: imgseq) else {
            return nil
        }
        metrics.recordGVHMR(CFAbsoluteTimeGetCurrent() - gvhmrStart)

        let lastIdx = GVHMRConstants.windowSize - 1
        let predX = output.predX[lastIdx]
        let predCam = output.predCam[lastIdx]

        var result = smplDecoder.decode(predX: predX, predCam: predCam, imageSize: imageSize)
        result.translFullCam = smplDecoder.computeTranslFullCam(
            predCam: predCam,
            bbxXYS: bbxXYS,
            focalLength: focalLength,
            imageSize: imageSize
        )

        if inference.smplReady {
            let bodyPoseFlat = result.bodyPoseAA.flatMap { [$0.x, $0.y, $0.z] }
            let orientFlat = [result.globalOrient.x, result.globalOrient.y, result.globalOrient.z]
            let smplStart = CFAbsoluteTimeGetCurrent()
            if let incamVerts = inference.runSMPLIncam(
                bodyPoseAA: bodyPoseFlat,
                globalOrientAA: orientFlat,
                betas: result.betas
            ) {
                metrics.recordSMPL(CFAbsoluteTimeGetCurrent() - smplStart)
                result.meshVerticesIncam = incamVerts
                result.meshVertices = incamVerts.map { SIMD3<Float>($0.x, -$0.y, -$0.z) }
            }
        }

        metrics.recordPipeline(CFAbsoluteTimeGetCurrent() - pipelineStart)

        return PersonFrameResult(
            personIndex: 0,
            trackID: trackID,
            keypoints: keypoints,
            bbox: bbox,
            bbxXYS: bbxXYS,
            gvhmrResult: result,
            predX: predX
        )
    }

    private func updateFPS() {
        frameCount += 1
        let now = Date()
        let elapsed = now.timeIntervalSince(lastFPSTime)
        if elapsed >= 1.0 {
            let currentFPS = Double(frameCount) / elapsed
            DispatchQueue.main.async { [weak self] in
                self?.fps = currentFPS
            }
            frameCount = 0
            lastFPSTime = now
        }
    }

    private func updateLiveCaption(
        result: GVHMRResult,
        personCount: Int,
        timestampSec: Double,
        pixelBuffer: CVPixelBuffer?
    ) {
        guard captionEnabled else {
            return
        }

        let useVisualCaption = (captionMode == .captioner) && captionTexter.isReady

        if useVisualCaption,
           captionFrameIndex % captionVisualStride == 0,
           let pixelBuffer {
            lastVisualCaption = captionTexter.caption(pixelBuffer: pixelBuffer)
            lastVisualCaptionFrame = captionFrameIndex
        }

        let visualToUse: String?
        if useVisualCaption,
           lastVisualCaptionFrame >= 0,
           (captionFrameIndex - lastVisualCaptionFrame) <= (captionVisualStride * 2) {
            visualToUse = lastVisualCaption
        } else {
            visualToUse = nil
        }

        let semantic = captionFusion.analyze(
            result: result,
            frameIndex: captionFrameIndex,
            timestampSec: timestampSec,
            personCount: max(personCount, 1),
            visualCaption: visualToUse
        )
        captionFrameIndex += 1

        DispatchQueue.main.async { [weak self] in
            self?.captionText = semantic.caption
            self?.captionConfidence = semantic.confidence
        }
    }

    private var personCaptionFramesCache: [Int: CaptionSemanticFrame] = [:]

    private func updateLiveCaptionForPerson(
        personResult: PersonFrameResult,
        personCount: Int,
        timestampSec: Double,
        pixelBuffer: CVPixelBuffer,
        imageSize: CGSize
    ) {
        guard captionEnabled else { return }

        let useVisualCaption = (captionMode == .captioner) && captionTexter.isReady

        let trackID = personResult.trackID
        let fusion = perPersonCaptionFusion[trackID] ?? {
            let f = CaptionFusionEngine()
            perPersonCaptionFusion[trackID] = f
            return f
        }()

        let frameIdx = perPersonCaptionFrameIndex[trackID] ?? 0

        if useVisualCaption,
           frameIdx % captionVisualStride == 0,
           let visual = captionTexter.caption(
                pixelBuffer: pixelBuffer,
                bbox: personResult.bbox,
                imageSize: imageSize
           ) {
            perPersonVisualCaption[trackID] = visual
            perPersonVisualCaptionFrame[trackID] = frameIdx
        }

        let visualToUse: String?
          if useVisualCaption,
              let seen = perPersonVisualCaptionFrame[trackID],
           (frameIdx - seen) <= (captionVisualStride * 2) {
            visualToUse = perPersonVisualCaption[trackID]
        } else {
            visualToUse = nil
        }

        let semantic = fusion.analyze(
            result: personResult.gvhmrResult,
            frameIndex: frameIdx,
            timestampSec: timestampSec,
            personCount: max(personCount, 1),
            visualCaption: visualToUse
        )

        personCaptionFramesCache[trackID] = semantic
        perPersonCaptionFrameIndex[trackID] = frameIdx + 1
    }
}
