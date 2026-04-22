// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Calvin Chan

import AVFoundation
import Combine
import CoreImage
import UIKit
import Vision
import SwiftUI

/// Processes a video file offline through the full GVHMR pipeline.
/// Reads all frames, runs 2D pose detection + feature extraction + GVHMR inference,
/// then stores per-frame results for rendering and export.
class VideoProcessor: ObservableObject {

    // MARK: - Published State

    @Published var progress: Float = 0       // 0..1
    @Published var phase: Phase = .idle
    @Published var frameResults: [FrameResult] = []
    @Published var multiPersonResults: [MultiPersonFrameResult] = []
    @Published var comparisonFrameResults: [GVHMRModelChoice: [FrameResult]] = [:]
    @Published var comparisonMultiPersonResults: [GVHMRModelChoice: [MultiPersonFrameResult]] = [:]
    @Published var videoSize: CGSize = .zero
    @Published var fps: Double = 30
    @Published var currentFrame: Int = 0
    @Published var error: String?
    @Published var isMultiPerson: Bool = false
    @Published var personCount: Int = 0
    @Published var captionTimeline: [CaptionSemanticFrame] = []
    @Published var captionSummary: String = ""
    @Published var multiPersonCaptionTimelines: [Int: [CaptionSemanticFrame]] = [:]

    enum Phase: String {
        case idle = "Ready"
        case loading = "Loading video..."
        case detecting = "Detecting poses..."
        case extracting = "Extracting features..."
        case inferring = "Running GVHMR..."
        case smpl = "Computing SMPL meshes..."
        case rendering = "Rendering output..."
        case done = "Done"
        case failed = "Failed"
    }

    /// Per-frame result from offline processing.
    struct FrameResult {
        let frameIndex: Int
        let keypoints: [SIMD3<Float>]     // 17 COCO keypoints in pixels
        let bbox: CGRect
        let bbxXYS: SIMD3<Float>          // (center_x, center_y, size)
        let gvhmrResult: GVHMRResult
        let predX: [Float]               // raw normalized 151-dim
    }

    private let inference = GVHMRInference()
    private let poseDetector = PoseDetector()
    private let vitPose = VitPoseProcessor()
    private let yoloPose = YOLOPoseProcessor()
    private let smplDecoder = SMPLDecoder()
    private let personTracker = PersonTracker()
    private let captionFusion = CaptionFusionEngine()
    private let processingQueue = DispatchQueue(label: "com.gvhmr.video", qos: .userInitiated)
    private var isCancelled = false

    var preprocessingMode: PreprocessingMode = .visionViTPose
    var maxPersons: Int = 5

    var smplFaces: [UInt32] = []
    var sourceVideoURL: URL?
    var focalLength: Float = 0
    private var frameImageGenerator: AVAssetImageGenerator?
    private var frameImageCache: [Int: UIImage] = [:]
    private let frameExtractQueue = DispatchQueue(label: "com.gvhmr.frameExtract")
    private let framePrefetchQueue = DispatchQueue(label: "com.gvhmr.framePrefetch", qos: .utility)
    private var pendingFrameRequest: Int?
    private var prefetchingFrames = Set<Int>()
    private var lastPrefetchTime: CFAbsoluteTime = 0
    private var frameDecodeGeneration: Int = 0

    let metrics = PerformanceMetrics()
    private var metricsCancellable: AnyCancellable?
    private let liveActivity = VideoProcessingLiveActivityManager.shared
    private var backgroundTaskID: UIBackgroundTaskIdentifier = .invalid
    private var isInBackground = false
    private var lastProgressPublished: Float = -1
    private var lastProgressPublishTime: CFAbsoluteTime = 0
    private var lastLiveActivityPublishTime: CFAbsoluteTime = 0
    private var lastLiveActivityProgress: Float = -1
    private var lastLiveActivityPhase: Phase = .idle
    private var lastComparisonProgressPublish: Float = -1
    private var lastComparisonProgressTime: CFAbsoluteTime = 0
    private var processingStartTime: CFAbsoluteTime = 0
    private var estimatedTotalFrames: Int = 0
    private var liveActivityProcessedFrames: Int = 0

    // MARK: - Setup

    func loadModels() {
        inference.loadModels()
        vitPose.loadModel()
        yoloPose.loadModel()
        smplFaces = SMPLMeshData.loadFaces()
        // Forward metrics changes so SwiftUI observes nested ObservableObject
        metricsCancellable = metrics.objectWillChange.sink { [weak self] _ in
            self?.objectWillChange.send()
        }
    }

    func handleScenePhase(_ scenePhase: ScenePhase) {
        switch scenePhase {
        case .background:
            isInBackground = true
            if isProcessing {
                beginBackgroundTask()
                pushLiveActivityUpdate(force: true)
            }
        case .active:
            isInBackground = false
            liveActivity.clearCompletedIfNeeded()
            endBackgroundTaskIfNeeded()
        case .inactive:
            break
        @unknown default:
            break
        }
    }

    private var isProcessing: Bool {
        phase != .idle && phase != .done && phase != .failed
    }

    /// Switch which GVHMR model variant is used for inference.
    func selectModel(_ choice: GVHMRModelChoice) {
        inference.selectModel(choice)
    }

    var isReady: Bool { inference.isReady }

    // MARK: - Frame Extraction

    func setupFrameExtractor() {
        guard let url = sourceVideoURL else { return }
        let asset = AVURLAsset(url: url)
        let gen = AVAssetImageGenerator(asset: asset)
        gen.appliesPreferredTrackTransform = true
        gen.requestedTimeToleranceBefore = .zero
        gen.requestedTimeToleranceAfter = .zero
        self.frameImageGenerator = gen
        self.frameImageCache = [:]
        self.prefetchingFrames = []
        self.pendingFrameRequest = nil
        self.frameDecodeGeneration += 1
        self.lastPrefetchTime = 0
    }

    func getFrameImage(at index: Int, completion: @escaping (UIImage?) -> Void) {
        if let cached = frameImageCache[index] {
            completion(cached)
            return
        }
        guard frameImageGenerator != nil else {
            completion(nil)
            return
        }
        // Serialize requests — only process the latest one
        pendingFrameRequest = index
        let generation = frameDecodeGeneration
        frameExtractQueue.async { [weak self] in
            guard let self = self,
                  let gen = self.frameImageGenerator,
                  generation == self.frameDecodeGeneration,
                  self.pendingFrameRequest == index else { return }
            let time = CMTime(seconds: Double(index) / max(self.fps, 1), preferredTimescale: 600)
            let cgImage = try? gen.copyCGImage(at: time, actualTime: nil)
            let image = cgImage.map { UIImage(cgImage: $0) }
            DispatchQueue.main.async {
                guard generation == self.frameDecodeGeneration else { return }
                if let img = image {
                    self.frameImageCache[index] = img
                    if self.frameImageCache.count > 80 {
                        let toRemove = self.frameImageCache.keys.filter { abs($0 - index) > 24 }
                        for k in toRemove { self.frameImageCache.removeValue(forKey: k) }
                    }
                }
                completion(image)
            }
        }
    }

    /// Prefetch nearby frames to reduce playback stalls during scrub/play.
    func prefetchFrames(around index: Int, radius: Int = 8) {
        guard frameImageGenerator != nil else { return }

        // Throttle prefetch to avoid queue buildup during active playback.
        let now = CFAbsoluteTimeGetCurrent()
        if now - lastPrefetchTime < 0.03 { return }
        lastPrefetchTime = now

        let totalFrames = max(
            frameResults.count,
            comparisonFrameResults[.small]?.count ?? 0,
            comparisonFrameResults[.medium]?.count ?? 0,
            comparisonFrameResults[.original]?.count ?? 0
        )
        guard totalFrames > 0 else { return }

        let start = max(0, index - radius)
        let end = min(totalFrames - 1, index + radius)
        let indices = (start...end).filter { frameImageCache[$0] == nil && !prefetchingFrames.contains($0) }
        if indices.isEmpty { return }

        for idx in indices { prefetchingFrames.insert(idx) }

        let generation = frameDecodeGeneration

        framePrefetchQueue.async { [weak self] in
            guard let self = self, let gen = self.frameImageGenerator else { return }
            for idx in indices {
                if generation != self.frameDecodeGeneration { return }
                let time = CMTime(seconds: Double(idx) / max(self.fps, 1), preferredTimescale: 600)
                let cgImage = try? gen.copyCGImage(at: time, actualTime: nil)
                let image = cgImage.map { UIImage(cgImage: $0) }
                DispatchQueue.main.async {
                    defer { self.prefetchingFrames.remove(idx) }
                    guard generation == self.frameDecodeGeneration else { return }
                    if let img = image {
                        self.frameImageCache[idx] = img
                        if self.frameImageCache.count > 120 {
                            let toRemove = self.frameImageCache.keys.filter { abs($0 - index) > 30 }
                            for k in toRemove { self.frameImageCache.removeValue(forKey: k) }
                        }
                    }
                }
            }
        }
    }

    // MARK: - Process Video

    func processVideo(url: URL) {
        sourceVideoURL = url
        beginBackgroundTask()
        DispatchQueue.main.async { [weak self] in
            self?.isMultiPerson = false
            self?.personCount = 0
            self?.multiPersonResults = []
            self?.multiPersonCaptionTimelines = [:]
        }
        processingQueue.async { [weak self] in
            self?.runFullPipeline(url: url)
        }
    }

    func cancel() {
        isCancelled = true
        DispatchQueue.main.async { [weak self] in
            self?.phase = .idle
            self?.progress = 0
            self?.frameImageGenerator = nil
            self?.frameImageCache = [:]
            self?.prefetchingFrames = []
            self?.pendingFrameRequest = nil
            self?.frameDecodeGeneration += 1
            self?.captionTimeline = []
            self?.captionSummary = ""
            self?.multiPersonCaptionTimelines = [:]
            self?.multiPersonResults = []
            self?.isMultiPerson = false
            self?.personCount = 0
            self?.metrics.isComparing = false
            self?.metrics.comparisonProgress = 0
            self?.metrics.comparisonPhase = "Cancelled"
            self?.estimatedTotalFrames = 0
            self?.liveActivityProcessedFrames = 0
            self?.currentFrame = 0
            self?.liveActivity.end(finalPhase: "Cancelled", progress: 0)
            self?.endBackgroundTaskIfNeeded()
        }
    }

    // MARK: - Full Pipeline

    private func runFullPipeline(url: URL) {
        isCancelled = false
        processingStartTime = CFAbsoluteTimeGetCurrent()
        lastProgressPublished = -1
        lastProgressPublishTime = 0
        lastLiveActivityPublishTime = 0
        lastLiveActivityProgress = -1
        lastLiveActivityPhase = .idle
        updatePhase(.loading)
        DispatchQueue.main.async { [weak self] in
            self?.isMultiPerson = false
            self?.personCount = 0
            self?.multiPersonResults = []
            self?.captionTimeline = []
            self?.captionSummary = ""
            self?.multiPersonCaptionTimelines = [:]
        }

        // 1. Load video metadata
        let asset = AVURLAsset(url: url)
        guard let track = asset.tracks(withMediaType: .video).first else {
            fail("No video track found")
            return
        }

        let size = track.naturalSize.applying(track.preferredTransform)
        let videoW = abs(size.width)
        let videoH = abs(size.height)
        let frameRate = Double(track.nominalFrameRate)
        let duration = CMTimeGetSeconds(asset.duration)
        let totalFrames = Int(duration * frameRate)
        estimatedTotalFrames = totalFrames
        liveActivityProcessedFrames = 0

        DispatchQueue.main.async { [weak self] in
            self?.videoSize = CGSize(width: videoW, height: videoH)
            self?.fps = frameRate
        }

        print("[VideoProcessor] Video: \(Int(videoW))x\(Int(videoH)), \(totalFrames) frames @ \(frameRate) fps")

        // 2. Read all frames and detect poses
        updatePhase(.detecting)

        guard let reader = try? AVAssetReader(asset: asset) else {
            fail("Cannot create asset reader")
            return
        }

        let outputSettings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: outputSettings)
        output.alwaysCopiesSampleData = false
        reader.add(output)
        reader.startReading()

        // Estimate focal length from video dimensions (default FoV ~60°)
        let fov: Float = 60.0
        let fl = Float(videoW) / (2.0 * tan(fov * .pi / 360.0))
        self.focalLength = fl
        let ppx = Float(videoW) / 2.0
        let ppy = Float(videoH) / 2.0
        let imgSize = CGSize(width: CGFloat(videoW), height: CGFloat(videoH))

        // Reset tracking state for new video
        poseDetector.resetTracking()
        yoloPose.resetTracking()

        // Phase 2a: Detect poses in all frames
        struct PreprocessedFrame {
            let index: Int
            let keypoints: [SIMD3<Float>]
            let normalizedKP: [SIMD3<Float>]
            let bbox: CGRect
            let bbxXYS: SIMD3<Float>
            let cliffCam: SIMD3<Float>
        }

        var preprocessed = [PreprocessedFrame]()
        var frameIdx = 0
        let useYOLO = preprocessingMode == .yoloPose && yoloPose.isReady
        let useHybrid = preprocessingMode == .yoloViTPose && yoloPose.isReady

        while let sampleBuffer = output.copyNextSampleBuffer(),
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {

            guard !isCancelled else { return } // cancelled

            // autoreleasepool prevents Vision/CoreML autorelease objects from accumulating
            // across hundreds of frames, which otherwise causes OOM crashes on long videos.
            autoreleasepool {
                let bbox: CGRect
                let keypoints: [SIMD3<Float>]

                if useYOLO {
                    // YOLO-Pose: single pass for bbox + keypoints
                    if let yoloResult = yoloPose.detect(pixelBuffer: pixelBuffer, imageSize: imgSize) {
                        bbox = yoloResult.bbox
                        keypoints = yoloResult.keypoints
                    } else {
                        bbox = CGRect(x: 0, y: 0, width: CGFloat(videoW), height: CGFloat(videoH))
                        keypoints = [SIMD3<Float>](repeating: .zero, count: 17)
                    }
                } else if useHybrid {
                    // Hybrid: YOLO for fast bbox + ViTPose for quality keypoints (2 inferences)
                    let yoloResult = yoloPose.detect(pixelBuffer: pixelBuffer, imageSize: imgSize)
                    bbox = yoloResult?.bbox ?? CGRect(x: 0, y: 0, width: CGFloat(videoW), height: CGFloat(videoH))

                    let bbxXYS_vit = poseDetector.computeBBXXYS(bbox: bbox)
                    if vitPose.isReady,
                       let vitKP = vitPose.extractKeypoints(
                           pixelBuffer: pixelBuffer, bbxXYS: bbxXYS_vit, imageSize: imgSize) {
                        keypoints = vitKP
                    } else {
                        keypoints = yoloResult?.keypoints ?? [SIMD3<Float>](repeating: .zero, count: 17)
                    }
                } else {
                    // Vision + ViTPose: two-step pipeline (3 inferences)
                    let detection = poseDetector.detect(pixelBuffer: pixelBuffer, imageSize: imgSize)

                    if let det = detection {
                        bbox = det.bbox
                    } else {
                        bbox = CGRect(x: 0, y: 0, width: CGFloat(videoW), height: CGFloat(videoH))
                    }

                    let bbxXYS_vit = poseDetector.computeBBXXYS(bbox: bbox)

                    if vitPose.isReady,
                       let vitKP = vitPose.extractKeypoints(
                           pixelBuffer: pixelBuffer, bbxXYS: bbxXYS_vit, imageSize: imgSize) {
                        keypoints = vitKP
                    } else if let det = detection {
                        keypoints = det.keypoints
                    } else {
                        keypoints = [SIMD3<Float>](repeating: .zero, count: 17)
                    }
                }

                let bbxXYS = poseDetector.computeBBXXYS(bbox: bbox)
                let normalizedKP = poseDetector.normalizeKeypoints(keypoints, bbxXYS: bbxXYS)
                let cliffCam = SIMD3<Float>(
                    (bbxXYS.x - ppx) / fl,
                    (bbxXYS.y - ppy) / fl,
                    bbxXYS.z / fl
                )

                preprocessed.append(PreprocessedFrame(
                    index: frameIdx,
                    keypoints: keypoints,
                    normalizedKP: normalizedKP,
                    bbox: bbox,
                    bbxXYS: bbxXYS,
                    cliffCam: cliffCam
                ))

                frameIdx += 1
                liveActivityProcessedFrames = min(frameIdx, totalFrames)
                updateProgress(Float(frameIdx) / Float(max(totalFrames, 1)) * 0.3)
            }
        }

        let numFrames = preprocessed.count
        print("[VideoProcessor] Detected poses in \(numFrames) frames")

        // Skip MobileNetProxy feature extraction — ablation testing showed that
        // the distilled proxy features (cos_sim ~0.78 with ViT on out-of-distribution video)
        // actually *hurt* the model more than passing zero features.
        // Zero features give BP_L2=1.20 / corr=0.84 vs proxy BP_L2=1.19 / GO_L2=3.25.
        // This also makes processing faster (skip entire second video pass).
        let zeroFeatures = [Float](repeating: 0, count: GVHMRConstants.imgseqDim)
        print("[VideoProcessor] Using zero image features (proxy features degrade quality)")

        // Phase 3: Run GVHMR inference over temporal windows
        updatePhase(.inferring)

        // For offline video, cam_angvel is identity (no gyroscope data)
        // Normalized identity: (angvel - mean) / std with mean=[1,0,0,0,1,0], std=[0.001,0.1,...]
        // = [0, 0, 0, 0, 0, 0]
        let zeroCamAngvel = [Float](repeating: 0, count: 6)

        let W = GVHMRConstants.windowSize
        var allPredX = [[Float]](repeating: [Float](repeating: 0, count: GVHMRConstants.outputDim), count: numFrames)
        var allPredCam = [SIMD3<Float>](repeating: .zero, count: numFrames)

        // Sliding window — use half-window stride for good quality with reasonable speed
        let stride = max(W / 2, 1)
        let halfW = W / 2

        // First, process windows at stride intervals
        var processedIndices = Set<Int>()
        var centerFrame = 0
        while centerFrame < numFrames {
            guard !isCancelled else { return }

            // Build window centered on centerFrame
            var windowObs = [Float]()
            var windowCliff = [Float]()
            var windowAngvel = [Float]()
            var windowImgseq = [Float]()

            for offset in 0..<W {
                let idx = min(max(centerFrame - halfW + offset, 0), numFrames - 1)
                for kp in preprocessed[idx].normalizedKP {
                    windowObs.append(kp.x)
                    windowObs.append(kp.y)
                    windowObs.append(kp.z)
                }
                windowCliff.append(preprocessed[idx].cliffCam.x)
                windowCliff.append(preprocessed[idx].cliffCam.y)
                windowCliff.append(preprocessed[idx].cliffCam.z)
                windowAngvel.append(contentsOf: zeroCamAngvel)
                windowImgseq.append(contentsOf: zeroFeatures)
            }

            if let result = inference.runGVHMR(
                obs: windowObs, cliffCam: windowCliff,
                camAngvel: windowAngvel, imgseq: windowImgseq
            ) {
                // Assign output for all frames in this window that haven't been assigned yet
                for offset in 0..<W {
                    let frameInVideo = centerFrame - halfW + offset
                    if frameInVideo >= 0 && frameInVideo < numFrames && !processedIndices.contains(frameInVideo) {
                        allPredX[frameInVideo] = result.predX[offset]
                        allPredCam[frameInVideo] = result.predCam[offset]
                        processedIndices.insert(frameInVideo)
                    }
                }
            }

            centerFrame += stride
            liveActivityProcessedFrames = min(processedIndices.count, totalFrames)
            updateProgress(0.3 + Float(processedIndices.count) / Float(numFrames) * 0.4)
        }

        print("[VideoProcessor] GVHMR inference complete")

        // Global betas averaging: the full-sequence model averages betas across
        // the entire sequence, but sliding windows produce per-window betas.
        // Average them globally to match full-sequence behavior.
        let betaStart = 126
        let betaCount = 10
        var globalBetas = [Float](repeating: 0, count: betaCount)
        for i in 0..<numFrames {
            for b in 0..<betaCount {
                globalBetas[b] += allPredX[i][betaStart + b]
            }
        }
        for b in 0..<betaCount {
            globalBetas[b] /= Float(numFrames)
        }
        for i in 0..<numFrames {
            for b in 0..<betaCount {
                allPredX[i][betaStart + b] = globalBetas[b]
            }
        }

        // Phase 4: Decode + SMPL mesh
        updatePhase(.smpl)

        var results = [FrameResult]()
        results.reserveCapacity(numFrames)

        for i in 0..<numFrames {
            guard !isCancelled else { return }

            var gvhmrResult = smplDecoder.decode(
                predX: allPredX[i],
                predCam: allPredCam[i],
                imageSize: imgSize
            )

            // Compute full-camera translation for perspective projection
            gvhmrResult.translFullCam = smplDecoder.computeTranslFullCam(
                predCam: allPredCam[i],
                bbxXYS: preprocessed[i].bbxXYS,
                focalLength: fl,
                imageSize: imgSize
            )

            // Run SMPL mesh
            if inference.smplReady {
                let bodyPoseFlat = gvhmrResult.bodyPoseAA.flatMap { [$0.x, $0.y, $0.z] }
                let orientFlat = [gvhmrResult.globalOrient.x, gvhmrResult.globalOrient.y, gvhmrResult.globalOrient.z]

                // Get vertices in camera coords (Y-down, Z-forward) for incam overlay
                if let incamVerts = inference.runSMPLIncam(
                    bodyPoseAA: bodyPoseFlat,
                    globalOrientAA: orientFlat,
                    betas: gvhmrResult.betas
                ) {
                    gvhmrResult.meshVerticesIncam = incamVerts
                    // Derive SceneKit vertices (Y-up, Z-toward-viewer) by flipping Y and Z
                    gvhmrResult.meshVertices = incamVerts.map { SIMD3<Float>($0.x, -$0.y, -$0.z) }
                }
            }

            results.append(FrameResult(
                frameIndex: i,
                keypoints: preprocessed[i].keypoints,
                bbox: preprocessed[i].bbox,
                bbxXYS: preprocessed[i].bbxXYS,
                gvhmrResult: gvhmrResult,
                predX: allPredX[i]
            ))

            liveActivityProcessedFrames = min(i + 1, totalFrames)
            updateProgress(0.7 + Float(i) / Float(numFrames) * 0.3)
        }

        // Build semantic timeline from decoded GVHMR outputs.
        captionFusion.reset()
        var timeline = [CaptionSemanticFrame]()
        timeline.reserveCapacity(results.count)
        for frame in results {
            let ts = Double(frame.frameIndex) / max(frameRate, 1.0)
            timeline.append(
                captionFusion.analyze(
                    result: frame.gvhmrResult,
                    frameIndex: frame.frameIndex,
                    timestampSec: ts,
                    personCount: 1
                )
            )
        }
        let summary = captionFusion.summarize(timeline)

        DispatchQueue.main.async { [weak self] in
            self?.isMultiPerson = false
            self?.personCount = 0
            self?.multiPersonResults = []
            self?.frameResults = results
            self?.comparisonFrameResults = [:]
            self?.captionTimeline = timeline
            self?.captionSummary = summary
            self?.multiPersonCaptionTimelines = [:]
            self?.phase = .done
            self?.progress = 1.0
            self?.setupFrameExtractor()
            self?.liveActivity.complete(finalPhase: "Done ✓", progress: 1.0)
            self?.endBackgroundTaskIfNeeded()
        }

        print("[VideoProcessor] Processing complete: \(results.count) frames")
    }

    // MARK: - Multi-Person Pipeline

    func processVideoMulti(url: URL) {
        sourceVideoURL = url
        beginBackgroundTask()
        processingQueue.async { [weak self] in
            self?.runMultiPersonPipeline(url: url)
        }
    }

    /// Multi-person video processing pipeline.
    /// Detects ALL persons per frame using YOLO, tracks them with IoU assignment,
    /// then runs GVHMR independently per person (same as Python demo_multi.py).
    private func runMultiPersonPipeline(url: URL) {
        isCancelled = false
        processingStartTime = CFAbsoluteTimeGetCurrent()
        lastProgressPublished = -1
        lastProgressPublishTime = 0
        lastLiveActivityPublishTime = 0
        lastLiveActivityProgress = -1
        lastLiveActivityPhase = .idle
        updatePhase(.loading)
        DispatchQueue.main.async { [weak self] in
            self?.captionTimeline = []
            self?.captionSummary = ""
            self?.multiPersonCaptionTimelines = [:]
        }

        DispatchQueue.main.async { [weak self] in
            self?.isMultiPerson = true
        }

        // 1. Load video metadata
        let asset = AVURLAsset(url: url)
        guard let track = asset.tracks(withMediaType: .video).first else {
            fail("No video track found")
            return
        }

        let size = track.naturalSize.applying(track.preferredTransform)
        let videoW = abs(size.width)
        let videoH = abs(size.height)
        let frameRate = Double(track.nominalFrameRate)
        let duration = CMTimeGetSeconds(asset.duration)
        let totalFrames = Int(duration * frameRate)
        estimatedTotalFrames = totalFrames
        liveActivityProcessedFrames = 0

        DispatchQueue.main.async { [weak self] in
            self?.videoSize = CGSize(width: videoW, height: videoH)
            self?.fps = frameRate
        }

        print("[VideoProcessor Multi] Video: \(Int(videoW))x\(Int(videoH)), \(totalFrames) frames @ \(frameRate) fps")

        // 2. Detect all persons in all frames
        updatePhase(.detecting)

        guard let reader = try? AVAssetReader(asset: asset) else {
            fail("Cannot create asset reader")
            return
        }

        let outputSettings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: outputSettings)
        output.alwaysCopiesSampleData = false
        reader.add(output)
        reader.startReading()

        let fov: Float = 60.0
        let fl = Float(videoW) / (2.0 * tan(fov * .pi / 360.0))
        self.focalLength = fl
        let ppx = Float(videoW) / 2.0
        let ppy = Float(videoH) / 2.0
        let imgSize = CGSize(width: CGFloat(videoW), height: CGFloat(videoH))

        personTracker.reset()

        // Per-frame, per-person raw detection data
        // Key: trackID → array of (frameIdx, keypoints, bbox)
        struct PersonDetection {
            let frameIdx: Int
            let keypoints: [SIMD3<Float>]
            let bbox: CGRect
        }

        var trackDetections = [Int: [PersonDetection]]()
        var allTrackIDs = Set<Int>()
        var frameIdx = 0

        while let sampleBuffer = output.copyNextSampleBuffer(),
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {

            guard !isCancelled else { return }

            autoreleasepool {
                // Detect all persons in this frame
                let detections = yoloPose.detectAll(pixelBuffer: pixelBuffer, imageSize: imgSize, maxPersons: maxPersons)

                // If using hybrid mode, refine keypoints with ViTPose for each person
                var finalDetections: [(keypoints: [SIMD3<Float>], bbox: CGRect, confidence: Float)]
                if preprocessingMode != .yoloPose && vitPose.isReady {
                    finalDetections = detections.map { det in
                        let bbxXYS = poseDetector.computeBBXXYS(bbox: det.bbox)
                        if let vitKP = vitPose.extractKeypoints(
                            pixelBuffer: pixelBuffer, bbxXYS: bbxXYS, imageSize: imgSize) {
                            return (keypoints: vitKP, bbox: det.bbox, confidence: det.confidence)
                        }
                        return det
                    }
                } else {
                    finalDetections = detections
                }

                // Track persons across frames
                let trackedPersons = personTracker.update(detections: finalDetections)

                // Store per-person detections
                for person in trackedPersons {
                    allTrackIDs.insert(person.trackID)
                    if trackDetections[person.trackID] == nil {
                        trackDetections[person.trackID] = []
                    }
                    trackDetections[person.trackID]!.append(PersonDetection(
                        frameIdx: frameIdx,
                        keypoints: person.keypoints,
                        bbox: person.bbox
                    ))
                }

                frameIdx += 1
                liveActivityProcessedFrames = min(frameIdx, totalFrames)
                updateProgress(Float(frameIdx) / Float(max(totalFrames, 1)) * 0.3)
            }
        }

        let numFrames = frameIdx
        let sortedTrackIDs = allTrackIDs.sorted()
        let numPersons = sortedTrackIDs.count

        DispatchQueue.main.async { [weak self] in
            self?.personCount = numPersons
        }

        print("[VideoProcessor Multi] Detected \(numPersons) person(s) across \(numFrames) frames")

        // 3. Run GVHMR independently per person
        updatePhase(.inferring)

        let zeroFeatures = [Float](repeating: 0, count: GVHMRConstants.imgseqDim)
        let zeroCamAngvel = [Float](repeating: 0, count: 6)
        let W = GVHMRConstants.windowSize
        let halfW = W / 2

        // Per-person GVHMR results: trackID → (frameIdx → (predX, predCam))
        struct PersonGVHMRResult {
            var predX: [Float]
            var predCam: SIMD3<Float>
        }
        var perPersonResults = [Int: [Int: PersonGVHMRResult]]()

        for (personIdx, trackID) in sortedTrackIDs.enumerated() {
            guard !isCancelled else { return }

            guard let detections = trackDetections[trackID], !detections.isEmpty else { continue }

            print("[VideoProcessor Multi] Running GVHMR for person \(personIdx) (trackID=\(trackID), \(detections.count) frames)")

            // Build per-person normalized data (only for frames where this person was detected)
            struct PersonFrame {
                let frameIdx: Int
                let normalizedKP: [SIMD3<Float>]
                let bbxXYS: SIMD3<Float>
                let cliffCam: SIMD3<Float>
                let keypoints: [SIMD3<Float>]
                let bbox: CGRect
            }

            var personFrames = [PersonFrame]()
            for det in detections {
                let bbxXYS = poseDetector.computeBBXXYS(bbox: det.bbox)
                let normalizedKP = poseDetector.normalizeKeypoints(det.keypoints, bbxXYS: bbxXYS)
                let cliffCam = SIMD3<Float>(
                    (bbxXYS.x - ppx) / fl,
                    (bbxXYS.y - ppy) / fl,
                    bbxXYS.z / fl
                )
                personFrames.append(PersonFrame(
                    frameIdx: det.frameIdx,
                    normalizedKP: normalizedKP,
                    bbxXYS: bbxXYS,
                    cliffCam: cliffCam,
                    keypoints: det.keypoints,
                    bbox: det.bbox
                ))
            }

            let N = personFrames.count
            var allPredX = [[Float]](repeating: [Float](repeating: 0, count: GVHMRConstants.outputDim), count: N)
            var allPredCam = [SIMD3<Float>](repeating: .zero, count: N)

            // Sliding window over this person's frames
            let stride = max(W / 2, 1)
            var processedIndices = Set<Int>()
            var centerFrame = 0

            while centerFrame < N {
                guard !isCancelled else { return }

                var windowObs = [Float]()
                var windowCliff = [Float]()
                var windowAngvel = [Float]()
                var windowImgseq = [Float]()

                for offset in 0..<W {
                    let idx = min(max(centerFrame - halfW + offset, 0), N - 1)
                    for kp in personFrames[idx].normalizedKP {
                        windowObs.append(kp.x)
                        windowObs.append(kp.y)
                        windowObs.append(kp.z)
                    }
                    windowCliff.append(personFrames[idx].cliffCam.x)
                    windowCliff.append(personFrames[idx].cliffCam.y)
                    windowCliff.append(personFrames[idx].cliffCam.z)
                    windowAngvel.append(contentsOf: zeroCamAngvel)
                    windowImgseq.append(contentsOf: zeroFeatures)
                }

                if let result = inference.runGVHMR(
                    obs: windowObs, cliffCam: windowCliff,
                    camAngvel: windowAngvel, imgseq: windowImgseq
                ) {
                    for offset in 0..<W {
                        let frameInPerson = centerFrame - halfW + offset
                        if frameInPerson >= 0 && frameInPerson < N && !processedIndices.contains(frameInPerson) {
                            allPredX[frameInPerson] = result.predX[offset]
                            allPredCam[frameInPerson] = result.predCam[offset]
                            processedIndices.insert(frameInPerson)
                        }
                    }
                }

                let personBase = Float(personIdx) / Float(max(numPersons, 1))
                let personInner = Float(processedIndices.count) / Float(max(N, 1))
                liveActivityProcessedFrames = min(max(0, Int((0.3 + (personBase + personInner / Float(max(numPersons, 1))) * 0.4) * Float(totalFrames))), totalFrames)
                updateProgress(0.3 + (personBase + personInner / Float(max(numPersons, 1))) * 0.4)

                centerFrame += stride
            }

            // Per-person betas averaging (not global — each person can have different body shape)
            let betaStart = 126
            let betaCount = 10
            var personBetas = [Float](repeating: 0, count: betaCount)
            for i in 0..<N {
                for b in 0..<betaCount {
                    personBetas[b] += allPredX[i][betaStart + b]
                }
            }
            for b in 0..<betaCount { personBetas[b] /= Float(N) }
            for i in 0..<N {
                for b in 0..<betaCount { allPredX[i][betaStart + b] = personBetas[b] }
            }

            // Store: map from video frame index → GVHMR result
            var frameMap = [Int: PersonGVHMRResult]()
            for (i, pf) in personFrames.enumerated() {
                frameMap[pf.frameIdx] = PersonGVHMRResult(predX: allPredX[i], predCam: allPredCam[i])
            }
            perPersonResults[trackID] = frameMap

            let personProgress = 0.3 + Float(personIdx + 1) / Float(numPersons) * 0.4
            updateProgress(personProgress)
        }

        print("[VideoProcessor Multi] GVHMR inference complete for all persons")

        // 4. SMPL decode + assemble multi-person frame results
        updatePhase(.smpl)

        var multiResults = [MultiPersonFrameResult]()
        multiResults.reserveCapacity(numFrames)
        // Also build single-person FrameResult array for backward compatibility (person 0)
        var singleResults = [FrameResult]()
        singleResults.reserveCapacity(numFrames)

        for frameI in 0..<numFrames {
            guard !isCancelled else { return }

            var persons = [PersonFrameResult]()

            for (personIdx, trackID) in sortedTrackIDs.enumerated() {
                guard let frameMap = perPersonResults[trackID],
                      let gvhmrRes = frameMap[frameI],
                      let detections = trackDetections[trackID],
                      let det = detections.first(where: { $0.frameIdx == frameI })
                else { continue }

                let bbxXYS = poseDetector.computeBBXXYS(bbox: det.bbox)

                var gvhmrResult = smplDecoder.decode(
                    predX: gvhmrRes.predX,
                    predCam: gvhmrRes.predCam,
                    imageSize: imgSize
                )

                gvhmrResult.translFullCam = smplDecoder.computeTranslFullCam(
                    predCam: gvhmrRes.predCam,
                    bbxXYS: bbxXYS,
                    focalLength: fl,
                    imageSize: imgSize
                )

                if inference.smplReady {
                    let bodyPoseFlat = gvhmrResult.bodyPoseAA.flatMap { [$0.x, $0.y, $0.z] }
                    let orientFlat = [gvhmrResult.globalOrient.x, gvhmrResult.globalOrient.y, gvhmrResult.globalOrient.z]
                    if let incamVerts = inference.runSMPLIncam(
                        bodyPoseAA: bodyPoseFlat,
                        globalOrientAA: orientFlat,
                        betas: gvhmrResult.betas
                    ) {
                        gvhmrResult.meshVerticesIncam = incamVerts
                        gvhmrResult.meshVertices = incamVerts.map { SIMD3<Float>($0.x, -$0.y, -$0.z) }
                    }
                }

                persons.append(PersonFrameResult(
                    personIndex: personIdx,
                    trackID: trackID,
                    keypoints: det.keypoints,
                    bbox: det.bbox,
                    bbxXYS: bbxXYS,
                    gvhmrResult: gvhmrResult,
                    predX: gvhmrRes.predX
                ))
            }

            multiResults.append(MultiPersonFrameResult(frameIndex: frameI, persons: persons))

            // Backward-compatible single-person result (first person or empty)
            if let first = persons.first {
                singleResults.append(FrameResult(
                    frameIndex: frameI,
                    keypoints: first.keypoints,
                    bbox: first.bbox,
                    bbxXYS: first.bbxXYS,
                    gvhmrResult: first.gvhmrResult,
                    predX: first.predX
                ))
            } else {
                let emptyResult = GVHMRResult(
                    joints3D: [SIMD3<Float>](repeating: .zero, count: 22),
                    joints2D: [CGPoint](repeating: .zero, count: 22),
                    predCam: .zero,
                    bodyPoseAA: [SIMD3<Float>](repeating: .zero, count: 21),
                    globalOrient: .zero,
                    betas: [Float](repeating: 0, count: 10),
                    confidence: 0
                )
                singleResults.append(FrameResult(
                    frameIndex: frameI,
                    keypoints: [],
                    bbox: .zero,
                    bbxXYS: .zero,
                    gvhmrResult: emptyResult,
                    predX: [Float](repeating: 0, count: GVHMRConstants.outputDim)
                ))
            }

            liveActivityProcessedFrames = min(frameI + 1, totalFrames)
            updateProgress(0.7 + Float(frameI) / Float(numFrames) * 0.3)
        }

        // Build frame-level semantic timeline from primary tracked person + person count.
        captionFusion.reset()
        var timeline = [CaptionSemanticFrame]()
        timeline.reserveCapacity(multiResults.count)
        for i in 0..<multiResults.count {
            let people = multiResults[i].persons
            let baseResult = people.first?.gvhmrResult ?? singleResults[i].gvhmrResult
            let ts = Double(i) / max(frameRate, 1.0)
            timeline.append(
                captionFusion.analyze(
                    result: baseResult,
                    frameIndex: i,
                    timestampSec: ts,
                    personCount: max(people.count, 1)
                )
            )
        }
        let summary = captionFusion.summarize(timeline)

        // Build per-track semantic timelines so UI can select person-specific captions.
        var perTrackTimelines: [Int: [CaptionSemanticFrame]] = [:]
        var perTrackFusion: [Int: CaptionFusionEngine] = [:]
        var perTrackFrameCount: [Int: Int] = [:]

        for frame in multiResults {
            let personCountInFrame = max(frame.persons.count, 1)
            for person in frame.persons {
                let fusion = perTrackFusion[person.trackID] ?? {
                    let f = CaptionFusionEngine()
                    perTrackFusion[person.trackID] = f
                    return f
                }()

                let localFrame = perTrackFrameCount[person.trackID, default: 0]
                let ts = Double(frame.frameIndex) / max(frameRate, 1.0)
                let semantic = fusion.analyze(
                    result: person.gvhmrResult,
                    frameIndex: frame.frameIndex,
                    timestampSec: ts,
                    personCount: personCountInFrame
                )
                perTrackTimelines[person.trackID, default: []].append(semantic)
                perTrackFrameCount[person.trackID] = localFrame + 1
            }
        }

        DispatchQueue.main.async { [weak self] in
            self?.multiPersonResults = multiResults
            self?.frameResults = singleResults
            self?.comparisonFrameResults = [:]
            self?.isMultiPerson = true
            self?.personCount = numPersons
            self?.captionTimeline = timeline
            self?.captionSummary = summary
            self?.multiPersonCaptionTimelines = perTrackTimelines
            self?.phase = .done
            self?.progress = 1.0
            self?.setupFrameExtractor()
            self?.liveActivity.complete(finalPhase: "Done ✓", progress: 1.0)
            self?.endBackgroundTaskIfNeeded()
        }

        print("[VideoProcessor Multi] Complete: \(numFrames) frames, \(numPersons) persons")
    }

    // MARK: - Helpers

    private func updatePhase(_ p: Phase) {
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.phase = p
            if p != .done && p != .failed && p != .idle {
                self.pushLiveActivityUpdate(force: true)
                if self.isInBackground {
                    self.beginBackgroundTask()
                }
            }
        }
    }

    private func updateProgress(_ p: Float) {
        let clamped = min(max(p, 0), 1)
        let now = CFAbsoluteTimeGetCurrent()
        if clamped < 1,
              clamped - lastProgressPublished < 0.01,
              now - lastProgressPublishTime < 0.10 {
            return
        }

        lastProgressPublished = clamped
        lastProgressPublishTime = now

        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.progress = clamped
            self.pushLiveActivityUpdate(force: false)
        }
    }

    private func liveActivityProgressValue() -> Float {
        metrics.isComparing ? metrics.comparisonProgress : progress
    }

    private func estimatedProcessedFrames() -> Int {
        guard estimatedTotalFrames > 0 else { return 0 }
        if liveActivityProcessedFrames > 0 {
            return min(estimatedTotalFrames, max(0, liveActivityProcessedFrames))
        }
        let p = min(max(liveActivityProgressValue(), 0), 1)
        return min(estimatedTotalFrames, Int((p * Float(estimatedTotalFrames)).rounded()))
    }

    private func liveActivityETASeconds() -> Int {
        let p = min(max(liveActivityProgressValue(), 0), 1)
        guard processingStartTime > 0, p > 0.01, p < 0.999 else { return 0 }
        let elapsed = CFAbsoluteTimeGetCurrent() - processingStartTime
        let remaining = (elapsed / Double(p)) - elapsed
        return max(0, Int(remaining.rounded()))
    }

    private func liveActivityDetailText() -> String {
        if let name = sourceVideoURL?.deletingPathExtension().lastPathComponent,
           !name.isEmpty {
            return name
        }
        return "GVHMR Video"
    }

    private func fail(_ message: String) {
        print("[VideoProcessor] ERROR: \(message)")
        DispatchQueue.main.async { [weak self] in
            self?.phase = .failed
            self?.error = message
            self?.estimatedTotalFrames = 0
            self?.liveActivityProcessedFrames = 0
            self?.currentFrame = 0
            self?.processingStartTime = 0
            self?.liveActivity.end(finalPhase: "Failed", progress: Double(self?.progress ?? 0))
            self?.endBackgroundTaskIfNeeded()
        }
    }

    private func pushLiveActivityUpdate(force: Bool) {
        guard isProcessing else { return }

        let now = CFAbsoluteTimeGetCurrent()
        if !force {
            let progressDelta = abs(progress - lastLiveActivityProgress)
            let phaseUnchanged = phase == lastLiveActivityPhase
            let tooSoon = now - lastLiveActivityPublishTime < 0.8
            if phaseUnchanged && progressDelta < 0.03 && tooSoon {
                return
            }
        }

        lastLiveActivityPublishTime = now
        lastLiveActivityProgress = progress
        lastLiveActivityPhase = phase

        let processed = estimatedProcessedFrames()
        let eta = liveActivityETASeconds()

        liveActivity.startOrUpdate(
            phase: phase.rawValue,
            detail: liveActivityDetailText(),
            progress: Double(progress),
            processedFrames: processed,
            totalFrames: estimatedTotalFrames,
            etaSeconds: eta,
            isMultiPerson: isMultiPerson,
            personCount: personCount,
            isCancellable: true
        )
    }

    private func beginBackgroundTask() {
        guard backgroundTaskID == .invalid else { return }
        backgroundTaskID = UIApplication.shared.beginBackgroundTask(withName: "GVHMRVideoProcessing") { [weak self] in
            self?.cancel()
            self?.endBackgroundTaskIfNeeded()
        }
    }

    private func endBackgroundTaskIfNeeded() {
        guard backgroundTaskID != .invalid else { return }
        UIApplication.shared.endBackgroundTask(backgroundTaskID)
        backgroundTaskID = .invalid
    }

    // MARK: - Model Comparison

    /// Run the same video through all available models back-to-back and benchmark each.
    func compareAllModels(url: URL) {
        sourceVideoURL = url
        beginBackgroundTask()
        processingQueue.async { [weak self] in
            self?.runComparison(url: url)
        }
    }

    private func runComparison(url: URL) {
        isCancelled = false
        processingStartTime = CFAbsoluteTimeGetCurrent()
        lastComparisonProgressPublish = -1
        lastComparisonProgressTime = 0

        DispatchQueue.main.async { [weak self] in
            self?.metrics.isComparing = true
            self?.metrics.comparisonResults = []
            self?.metrics.comparisonProgress = 0
            self?.comparisonFrameResults = [:]
            self?.comparisonMultiPersonResults = [:]
            self?.liveActivity.startOrUpdate(
                phase: "Comparing models...",
                detail: "Preparing benchmark",
                progress: 0,
                processedFrames: 0,
                totalFrames: 0,
                etaSeconds: 0,
                isMultiPerson: self?.isMultiPerson ?? false,
                personCount: self?.personCount ?? 0,
                isCancellable: true
            )
        }

        if isMultiPerson {
            runComparisonMulti(url: url)
            return
        }

        // 1. Load video and preprocess once (shared across all models)
        let asset = AVURLAsset(url: url)
        guard let track = asset.tracks(withMediaType: .video).first else {
            DispatchQueue.main.async { [weak self] in self?.metrics.isComparing = false }
            return
        }

        let size = track.naturalSize.applying(track.preferredTransform)
        let videoW = abs(size.width)
        let videoH = abs(size.height)
        let frameRate = Double(track.nominalFrameRate)
        let duration = CMTimeGetSeconds(asset.duration)
        let totalFrames = Int(duration * frameRate)
        estimatedTotalFrames = totalFrames
        liveActivityProcessedFrames = 0

        let fov: Float = 60.0
        let fl = Float(videoW) / (2.0 * tan(fov * .pi / 360.0))
        let ppx = Float(videoW) / 2.0
        let ppy = Float(videoH) / 2.0
        let imgSize = CGSize(width: CGFloat(videoW), height: CGFloat(videoH))

        // Publish geometry/intrinsics for compare preview rendering (incam projection).
        self.focalLength = fl
        DispatchQueue.main.async { [weak self] in
            self?.videoSize = imgSize
            self?.fps = frameRate
        }

        // Phase 1: Detect poses once
        updateComparisonPhase("Detecting poses...")

        poseDetector.resetTracking()
        yoloPose.resetTracking()

        struct PreprocessedFrame {
            let normalizedKP: [SIMD3<Float>]
            let bbxXYS: SIMD3<Float>
            let cliffCam: SIMD3<Float>
        }

        guard let reader = try? AVAssetReader(asset: asset) else {
            DispatchQueue.main.async { [weak self] in self?.metrics.isComparing = false }
            return
        }
        let outputSettings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: outputSettings)
        output.alwaysCopiesSampleData = false
        reader.add(output)
        reader.startReading()

        let detectStart = CFAbsoluteTimeGetCurrent()
        var preprocessed = [PreprocessedFrame]()
        var frameIdx = 0
        let useYOLO = preprocessingMode == .yoloPose && yoloPose.isReady
        let useHybrid = preprocessingMode == .yoloViTPose && yoloPose.isReady

        while let sampleBuffer = output.copyNextSampleBuffer(),
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
            guard !isCancelled else { return }

            autoreleasepool {
                let bbox: CGRect
                let keypoints: [SIMD3<Float>]

                if useYOLO {
                    if let yoloResult = yoloPose.detect(pixelBuffer: pixelBuffer, imageSize: imgSize) {
                        bbox = yoloResult.bbox
                        keypoints = yoloResult.keypoints
                    } else {
                        bbox = CGRect(x: 0, y: 0, width: CGFloat(videoW), height: CGFloat(videoH))
                        keypoints = [SIMD3<Float>](repeating: .zero, count: 17)
                    }
                } else if useHybrid {
                    let yoloResult = yoloPose.detect(pixelBuffer: pixelBuffer, imageSize: imgSize)
                    bbox = yoloResult?.bbox ?? CGRect(x: 0, y: 0, width: CGFloat(videoW), height: CGFloat(videoH))

                    let bbxXYS_vit = poseDetector.computeBBXXYS(bbox: bbox)
                    if vitPose.isReady,
                       let vitKP = vitPose.extractKeypoints(pixelBuffer: pixelBuffer, bbxXYS: bbxXYS_vit, imageSize: imgSize) {
                        keypoints = vitKP
                    } else {
                        keypoints = yoloResult?.keypoints ?? [SIMD3<Float>](repeating: .zero, count: 17)
                    }
                } else {
                    let detection = poseDetector.detect(pixelBuffer: pixelBuffer, imageSize: imgSize)
                    bbox = detection?.bbox ?? CGRect(x: 0, y: 0, width: CGFloat(videoW), height: CGFloat(videoH))

                    let bbxXYS_vit = poseDetector.computeBBXXYS(bbox: bbox)
                    if vitPose.isReady,
                       let vitKP = vitPose.extractKeypoints(pixelBuffer: pixelBuffer, bbxXYS: bbxXYS_vit, imageSize: imgSize) {
                        keypoints = vitKP
                    } else if let det = detection {
                        keypoints = det.keypoints
                    } else {
                        keypoints = [SIMD3<Float>](repeating: .zero, count: 17)
                    }
                }

                let bbxXYS = poseDetector.computeBBXXYS(bbox: bbox)
                let normalizedKP = poseDetector.normalizeKeypoints(keypoints, bbxXYS: bbxXYS)
                let cliffCam = SIMD3<Float>(
                    (bbxXYS.x - ppx) / fl,
                    (bbxXYS.y - ppy) / fl,
                    bbxXYS.z / fl
                )

                preprocessed.append(PreprocessedFrame(normalizedKP: normalizedKP, bbxXYS: bbxXYS, cliffCam: cliffCam))
                frameIdx += 1
                liveActivityProcessedFrames = min(frameIdx, totalFrames)
                let detectRatio = Float(frameIdx) / Float(max(totalFrames, 1))
                updateComparisonProgress(0.08 * detectRatio)
            }
        }

        let detectTime = CFAbsoluteTimeGetCurrent() - detectStart
        let numFrames = preprocessed.count

        print("[Compare] Preprocessed \(numFrames) frames in \(String(format: "%.1f", detectTime))s")

        // Phase 2: Run each model
        let availableModels = inference.availableModels
        let compareTotalFrames = max(numFrames * max(availableModels.count, 1), 1)
        estimatedTotalFrames = compareTotalFrames
        let zeroFeatures = [Float](repeating: 0, count: GVHMRConstants.imgseqDim)
        let zeroCamAngvel = [Float](repeating: 0, count: 6)
        let W = GVHMRConstants.windowSize
        let halfW = W / 2

        for (modelIdx, model) in availableModels.enumerated() {
            guard !isCancelled else { return }

            let modelCount = Float(max(availableModels.count, 1))
            let modelBase = 0.08 + (Float(modelIdx) / modelCount) * 0.92
            let modelSpan = 0.92 / modelCount

            updateComparisonPhase("Running \(model.rawValue)...")
            inference.selectModel(model)

            let peakMemBefore = PerformanceMetrics.currentMemoryMB()

            // GVHMR inference
            let gvhmrStart = CFAbsoluteTimeGetCurrent()
            var allPredX = [[Float]](repeating: [Float](repeating: 0, count: GVHMRConstants.outputDim), count: numFrames)
            var allPredCam = [SIMD3<Float>](repeating: .zero, count: numFrames)
            var processedIndices = Set<Int>()
            var centerFrame = 0

            while centerFrame < numFrames {
                guard !isCancelled else { return }

                var windowObs = [Float]()
                var windowCliff = [Float]()
                var windowAngvel = [Float]()
                var windowImgseq = [Float]()

                for offset in 0..<W {
                    let idx = min(max(centerFrame - halfW + offset, 0), numFrames - 1)
                    for kp in preprocessed[idx].normalizedKP {
                        windowObs.append(kp.x); windowObs.append(kp.y); windowObs.append(kp.z)
                    }
                    windowCliff.append(preprocessed[idx].cliffCam.x)
                    windowCliff.append(preprocessed[idx].cliffCam.y)
                    windowCliff.append(preprocessed[idx].cliffCam.z)
                    windowAngvel.append(contentsOf: zeroCamAngvel)
                    windowImgseq.append(contentsOf: zeroFeatures)
                }

                if let result = inference.runGVHMR(
                    obs: windowObs, cliffCam: windowCliff,
                    camAngvel: windowAngvel, imgseq: windowImgseq
                ) {
                    for offset in 0..<W {
                        let frameInVideo = centerFrame - halfW + offset
                        if frameInVideo >= 0 && frameInVideo < numFrames && !processedIndices.contains(frameInVideo) {
                            allPredX[frameInVideo] = result.predX[offset]
                            allPredCam[frameInVideo] = result.predCam[offset]
                            processedIndices.insert(frameInVideo)
                        }
                    }
                }

                let inferRatio = Float(processedIndices.count) / Float(max(numFrames, 1))
                liveActivityProcessedFrames = min(modelIdx * numFrames + processedIndices.count, compareTotalFrames)
                updateComparisonProgress(modelBase + modelSpan * 0.72 * inferRatio)
                centerFrame += max(W / 2, 1)
            }

            let gvhmrTime = CFAbsoluteTimeGetCurrent() - gvhmrStart

            // SMPL mesh
            let smplStart = CFAbsoluteTimeGetCurrent()
            var modelFrames = [FrameResult]()
            modelFrames.reserveCapacity(numFrames)
            if inference.smplReady {
                for i in 0..<numFrames {
                    guard !isCancelled else { return }
                    var decoded = smplDecoder.decode(predX: allPredX[i], predCam: allPredCam[i], imageSize: imgSize)
                    decoded.translFullCam = smplDecoder.computeTranslFullCam(
                        predCam: allPredCam[i],
                        bbxXYS: preprocessed[i].bbxXYS,
                        focalLength: fl,
                        imageSize: imgSize
                    )
                    let bodyPoseFlat = decoded.bodyPoseAA.flatMap { [$0.x, $0.y, $0.z] }
                    let orientFlat = [decoded.globalOrient.x, decoded.globalOrient.y, decoded.globalOrient.z]
                    if let incam = inference.runSMPLIncam(bodyPoseAA: bodyPoseFlat, globalOrientAA: orientFlat, betas: decoded.betas) {
                        decoded.meshVerticesIncam = incam
                        decoded.meshVertices = incam.map { SIMD3<Float>($0.x, -$0.y, -$0.z) }
                    }

                    // For compare previews we only need mesh/camera outputs.
                    modelFrames.append(FrameResult(
                        frameIndex: i,
                        keypoints: [],
                        bbox: .zero,
                        bbxXYS: preprocessed[i].bbxXYS,
                        gvhmrResult: decoded,
                        predX: allPredX[i]
                    ))

                    if i % 8 == 0 || i == numFrames - 1 {
                        liveActivityProcessedFrames = min(modelIdx * numFrames + (i + 1), compareTotalFrames)
                        let smplRatio = Float(i + 1) / Float(max(numFrames, 1))
                        updateComparisonProgress(modelBase + modelSpan * (0.72 + 0.23 * smplRatio))
                    }
                }
            } else {
                for i in 0..<numFrames {
                    var decoded = smplDecoder.decode(predX: allPredX[i], predCam: allPredCam[i], imageSize: imgSize)
                    decoded.translFullCam = smplDecoder.computeTranslFullCam(
                        predCam: allPredCam[i],
                        bbxXYS: preprocessed[i].bbxXYS,
                        focalLength: fl,
                        imageSize: imgSize
                    )
                    modelFrames.append(FrameResult(
                        frameIndex: i,
                        keypoints: [],
                        bbox: .zero,
                        bbxXYS: preprocessed[i].bbxXYS,
                        gvhmrResult: decoded,
                        predX: allPredX[i]
                    ))

                    if i % 8 == 0 || i == numFrames - 1 {
                        liveActivityProcessedFrames = min(modelIdx * numFrames + (i + 1), compareTotalFrames)
                        let smplRatio = Float(i + 1) / Float(max(numFrames, 1))
                        updateComparisonProgress(modelBase + modelSpan * (0.72 + 0.23 * smplRatio))
                    }
                }
            }
            let smplTime = CFAbsoluteTimeGetCurrent() - smplStart

            let peakMem = max(PerformanceMetrics.currentMemoryMB(), peakMemBefore)
            let totalTime = gvhmrTime + smplTime

            let benchmark = PerformanceMetrics.ModelBenchmark(
                model: model,
                totalTimeSec: totalTime,
                detectTimeSec: detectTime,
                gvhmrTimeSec: gvhmrTime,
                smplTimeSec: smplTime,
                numFrames: numFrames,
                avgGVHMRMs: (gvhmrTime / Double(numFrames)) * 1000,
                avgSMPLMs: numFrames > 0 ? (smplTime / Double(numFrames)) * 1000 : 0,
                peakMemoryMB: peakMem
            )

            print("[Compare] \(model.rawValue): total=\(String(format: "%.1f", totalTime))s, gvhmr=\(String(format: "%.1f", gvhmrTime))s, smpl=\(String(format: "%.1f", smplTime))s")

            DispatchQueue.main.async { [weak self] in
                self?.metrics.comparisonResults.append(benchmark)
                self?.comparisonFrameResults[model] = modelFrames
                self?.comparisonMultiPersonResults[model] = []
                self?.metrics.comparisonProgress = Float(modelIdx + 1) / Float(availableModels.count)
            }
        }

        DispatchQueue.main.async { [weak self] in
            self?.metrics.isComparing = false
            self?.metrics.comparisonProgress = 1.0
            self?.metrics.comparisonPhase = "Done"
            self?.liveActivity.complete(finalPhase: "Comparison done ✓", progress: 1.0)
            self?.endBackgroundTaskIfNeeded()
        }
    }

    private func updateComparisonPhase(_ phase: String) {
        DispatchQueue.main.async { [weak self] in
            self?.metrics.comparisonPhase = phase
            guard let self = self, self.metrics.isComparing else { return }
            self.liveActivity.startOrUpdate(
                phase: phase,
                detail: self.liveActivityDetailText(),
                progress: Double(self.metrics.comparisonProgress),
                processedFrames: self.estimatedProcessedFrames(),
                totalFrames: self.estimatedTotalFrames,
                etaSeconds: self.liveActivityETASeconds(),
                isMultiPerson: self.isMultiPerson,
                personCount: self.personCount,
                isCancellable: true
            )
        }
    }

    private func updateComparisonProgress(_ p: Float) {
        let clamped = min(max(p, 0), 1)
        let now = CFAbsoluteTimeGetCurrent()
        if clamped - lastComparisonProgressPublish < 0.004,
           now - lastComparisonProgressTime < 0.08 {
            return
        }
        lastComparisonProgressPublish = clamped
        lastComparisonProgressTime = now
        DispatchQueue.main.async { [weak self] in
            self?.metrics.comparisonProgress = clamped
            guard let self = self, self.metrics.isComparing else { return }
            self.liveActivity.startOrUpdate(
                phase: self.metrics.comparisonPhase,
                detail: self.liveActivityDetailText(),
                progress: Double(clamped),
                processedFrames: self.estimatedProcessedFrames(),
                totalFrames: self.estimatedTotalFrames,
                etaSeconds: self.liveActivityETASeconds(),
                isMultiPerson: self.isMultiPerson,
                personCount: self.personCount,
                isCancellable: true
            )
        }
    }

    /// Multi-person compare pipeline: preprocess once, then run all models over tracked persons.
    private func runComparisonMulti(url: URL) {
        lastComparisonProgressPublish = -1
        lastComparisonProgressTime = 0
        processingStartTime = CFAbsoluteTimeGetCurrent()

        let asset = AVURLAsset(url: url)
        guard let track = asset.tracks(withMediaType: .video).first else {
            DispatchQueue.main.async { [weak self] in self?.metrics.isComparing = false }
            return
        }

        let size = track.naturalSize.applying(track.preferredTransform)
        let videoW = abs(size.width)
        let videoH = abs(size.height)
        let frameRate = Double(track.nominalFrameRate)
        let duration = CMTimeGetSeconds(asset.duration)
        let totalFrames = Int(duration * frameRate)
        estimatedTotalFrames = totalFrames
        liveActivityProcessedFrames = 0

        let fov: Float = 60.0
        let fl = Float(videoW) / (2.0 * tan(fov * .pi / 360.0))
        let ppx = Float(videoW) / 2.0
        let ppy = Float(videoH) / 2.0
        let imgSize = CGSize(width: CGFloat(videoW), height: CGFloat(videoH))

        self.focalLength = fl
        DispatchQueue.main.async { [weak self] in
            self?.videoSize = imgSize
            self?.fps = frameRate
        }

        updateComparisonPhase("Detecting persons...")
        poseDetector.resetTracking()
        yoloPose.resetTracking()
        personTracker.reset()

        struct PersonDetection {
            let frameIdx: Int
            let keypoints: [SIMD3<Float>]
            let bbox: CGRect
        }

        guard let reader = try? AVAssetReader(asset: asset) else {
            DispatchQueue.main.async { [weak self] in self?.metrics.isComparing = false }
            return
        }
        let outputSettings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: outputSettings)
        output.alwaysCopiesSampleData = false
        reader.add(output)
        reader.startReading()

        var trackDetections = [Int: [PersonDetection]]()
        var allTrackIDs = Set<Int>()
        var frameIdx = 0
        let detectStart = CFAbsoluteTimeGetCurrent()

        while let sampleBuffer = output.copyNextSampleBuffer(),
              let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
            guard !isCancelled else { return }
            autoreleasepool {
                let detections = yoloPose.detectAll(pixelBuffer: pixelBuffer, imageSize: imgSize, maxPersons: maxPersons)

                var finalDetections: [(keypoints: [SIMD3<Float>], bbox: CGRect, confidence: Float)]
                if preprocessingMode != .yoloPose && vitPose.isReady {
                    finalDetections = detections.map { det in
                        let bbxXYS = poseDetector.computeBBXXYS(bbox: det.bbox)
                        if let vitKP = vitPose.extractKeypoints(pixelBuffer: pixelBuffer, bbxXYS: bbxXYS, imageSize: imgSize) {
                            return (keypoints: vitKP, bbox: det.bbox, confidence: det.confidence)
                        }
                        return det
                    }
                } else {
                    finalDetections = detections
                }

                let tracked = personTracker.update(detections: finalDetections)
                for person in tracked {
                    allTrackIDs.insert(person.trackID)
                    if trackDetections[person.trackID] == nil {
                        trackDetections[person.trackID] = []
                    }
                    trackDetections[person.trackID]!.append(PersonDetection(
                        frameIdx: frameIdx,
                        keypoints: person.keypoints,
                        bbox: person.bbox
                    ))
                }
                frameIdx += 1
                liveActivityProcessedFrames = min(frameIdx, totalFrames)
                let detectRatio = Float(frameIdx) / Float(max(totalFrames, 1))
                updateComparisonProgress(0.10 * detectRatio)
            }
        }

        let detectTime = CFAbsoluteTimeGetCurrent() - detectStart
        let numFrames = frameIdx
        let sortedTrackIDs = allTrackIDs.sorted()

        DispatchQueue.main.async { [weak self] in
            self?.personCount = sortedTrackIDs.count
        }

        let availableModels = inference.availableModels
        let compareTotalFrames = max(numFrames * max(availableModels.count, 1), 1)
        estimatedTotalFrames = compareTotalFrames
        let zeroFeatures = [Float](repeating: 0, count: GVHMRConstants.imgseqDim)
        let zeroCamAngvel = [Float](repeating: 0, count: 6)
        let W = GVHMRConstants.windowSize
        let halfW = W / 2

        struct PersonGVHMRResult {
            var predX: [Float]
            var predCam: SIMD3<Float>
        }

        for (modelIdx, model) in availableModels.enumerated() {
            guard !isCancelled else { return }
            let modelCount = Float(max(availableModels.count, 1))
            let modelBase = 0.10 + (Float(modelIdx) / modelCount) * 0.90
            let modelSpan = 0.90 / modelCount
            updateComparisonPhase("Running \(model.rawValue) (multi-person)...")
            inference.selectModel(model)

            let peakMemBefore = PerformanceMetrics.currentMemoryMB()

            let gvhmrStart = CFAbsoluteTimeGetCurrent()
            var perPersonResults = [Int: [Int: PersonGVHMRResult]]()

            for (trackIdx, trackID) in sortedTrackIDs.enumerated() {
                guard let detections = trackDetections[trackID], !detections.isEmpty else { continue }

                struct PersonFrame {
                    let frameIdx: Int
                    let normalizedKP: [SIMD3<Float>]
                    let bbxXYS: SIMD3<Float>
                    let cliffCam: SIMD3<Float>
                    let keypoints: [SIMD3<Float>]
                    let bbox: CGRect
                }

                var personFrames = [PersonFrame]()
                for det in detections {
                    let bbxXYS = poseDetector.computeBBXXYS(bbox: det.bbox)
                    let normalizedKP = poseDetector.normalizeKeypoints(det.keypoints, bbxXYS: bbxXYS)
                    let cliffCam = SIMD3<Float>(
                        (bbxXYS.x - ppx) / fl,
                        (bbxXYS.y - ppy) / fl,
                        bbxXYS.z / fl
                    )
                    personFrames.append(PersonFrame(
                        frameIdx: det.frameIdx,
                        normalizedKP: normalizedKP,
                        bbxXYS: bbxXYS,
                        cliffCam: cliffCam,
                        keypoints: det.keypoints,
                        bbox: det.bbox
                    ))
                }

                let N = personFrames.count
                var allPredX = [[Float]](repeating: [Float](repeating: 0, count: GVHMRConstants.outputDim), count: N)
                var allPredCam = [SIMD3<Float>](repeating: .zero, count: N)
                var processedIndices = Set<Int>()
                var centerFrame = 0

                while centerFrame < N {
                    var windowObs = [Float](); windowObs.reserveCapacity(W * 17 * 3)
                    var windowCliff = [Float](); windowCliff.reserveCapacity(W * 3)
                    var windowAngvel = [Float](); windowAngvel.reserveCapacity(W * 6)
                    var windowImgseq = [Float](); windowImgseq.reserveCapacity(W * GVHMRConstants.imgseqDim)

                    for offset in 0..<W {
                        let idx = min(max(centerFrame - halfW + offset, 0), N - 1)
                        for kp in personFrames[idx].normalizedKP {
                            windowObs.append(kp.x); windowObs.append(kp.y); windowObs.append(kp.z)
                        }
                        windowCliff.append(personFrames[idx].cliffCam.x)
                        windowCliff.append(personFrames[idx].cliffCam.y)
                        windowCliff.append(personFrames[idx].cliffCam.z)
                        windowAngvel.append(contentsOf: zeroCamAngvel)
                        windowImgseq.append(contentsOf: zeroFeatures)
                    }

                    if let result = inference.runGVHMR(obs: windowObs, cliffCam: windowCliff, camAngvel: windowAngvel, imgseq: windowImgseq) {
                        for offset in 0..<W {
                            let frameInPerson = centerFrame - halfW + offset
                            if frameInPerson >= 0 && frameInPerson < N && !processedIndices.contains(frameInPerson) {
                                allPredX[frameInPerson] = result.predX[offset]
                                allPredCam[frameInPerson] = result.predCam[offset]
                                processedIndices.insert(frameInPerson)
                            }
                        }
                    }

                    let perTrackRatio = Float(processedIndices.count) / Float(max(N, 1))
                    let trackRatio = (Float(trackIdx) + perTrackRatio) / Float(max(sortedTrackIDs.count, 1))
                    let modelFrameProgress = Int(trackRatio * Float(numFrames))
                    liveActivityProcessedFrames = min(modelIdx * numFrames + modelFrameProgress, compareTotalFrames)
                    updateComparisonProgress(modelBase + modelSpan * 0.70 * trackRatio)
                    centerFrame += max(W / 2, 1)
                }

                let betaStart = 126
                let betaCount = 10
                var personBetas = [Float](repeating: 0, count: betaCount)
                for i in 0..<N {
                    for b in 0..<betaCount { personBetas[b] += allPredX[i][betaStart + b] }
                }
                for b in 0..<betaCount { personBetas[b] /= Float(max(N, 1)) }
                for i in 0..<N {
                    for b in 0..<betaCount { allPredX[i][betaStart + b] = personBetas[b] }
                }

                var frameMap = [Int: PersonGVHMRResult]()
                for (i, pf) in personFrames.enumerated() {
                    frameMap[pf.frameIdx] = PersonGVHMRResult(predX: allPredX[i], predCam: allPredCam[i])
                }
                perPersonResults[trackID] = frameMap
            }

            let gvhmrTime = CFAbsoluteTimeGetCurrent() - gvhmrStart

            let smplStart = CFAbsoluteTimeGetCurrent()
            var multiResults = [MultiPersonFrameResult]()
            multiResults.reserveCapacity(numFrames)
            var singleResults = [FrameResult]()
            singleResults.reserveCapacity(numFrames)

            for frameI in 0..<numFrames {
                var persons = [PersonFrameResult]()

                for (personIdx, trackID) in sortedTrackIDs.enumerated() {
                    guard let frameMap = perPersonResults[trackID],
                          let gvhmrRes = frameMap[frameI],
                          let detections = trackDetections[trackID],
                          let det = detections.first(where: { $0.frameIdx == frameI })
                    else { continue }

                    let bbxXYS = poseDetector.computeBBXXYS(bbox: det.bbox)
                    var decoded = smplDecoder.decode(predX: gvhmrRes.predX, predCam: gvhmrRes.predCam, imageSize: imgSize)
                    decoded.translFullCam = smplDecoder.computeTranslFullCam(
                        predCam: gvhmrRes.predCam,
                        bbxXYS: bbxXYS,
                        focalLength: fl,
                        imageSize: imgSize
                    )

                    if inference.smplReady {
                        let bodyPoseFlat = decoded.bodyPoseAA.flatMap { [$0.x, $0.y, $0.z] }
                        let orientFlat = [decoded.globalOrient.x, decoded.globalOrient.y, decoded.globalOrient.z]
                        if let incam = inference.runSMPLIncam(bodyPoseAA: bodyPoseFlat, globalOrientAA: orientFlat, betas: decoded.betas) {
                            decoded.meshVerticesIncam = incam
                            decoded.meshVertices = incam.map { SIMD3<Float>($0.x, -$0.y, -$0.z) }
                        }
                    }

                    persons.append(PersonFrameResult(
                        personIndex: personIdx,
                        trackID: trackID,
                        keypoints: det.keypoints,
                        bbox: det.bbox,
                        bbxXYS: bbxXYS,
                        gvhmrResult: decoded,
                        predX: gvhmrRes.predX
                    ))
                }

                multiResults.append(MultiPersonFrameResult(frameIndex: frameI, persons: persons))

                if let first = persons.first {
                    singleResults.append(FrameResult(
                        frameIndex: frameI,
                        keypoints: first.keypoints,
                        bbox: first.bbox,
                        bbxXYS: first.bbxXYS,
                        gvhmrResult: first.gvhmrResult,
                        predX: first.predX
                    ))
                } else {
                    let emptyResult = GVHMRResult(
                        joints3D: [SIMD3<Float>](repeating: .zero, count: 22),
                        joints2D: [CGPoint](repeating: .zero, count: 22),
                        predCam: .zero,
                        bodyPoseAA: [SIMD3<Float>](repeating: .zero, count: 21),
                        globalOrient: .zero,
                        betas: [Float](repeating: 0, count: 10),
                        confidence: 0
                    )
                    singleResults.append(FrameResult(
                        frameIndex: frameI,
                        keypoints: [],
                        bbox: .zero,
                        bbxXYS: .zero,
                        gvhmrResult: emptyResult,
                        predX: [Float](repeating: 0, count: GVHMRConstants.outputDim)
                    ))
                }

                if frameI % 8 == 0 || frameI == numFrames - 1 {
                    liveActivityProcessedFrames = min(modelIdx * numFrames + (frameI + 1), compareTotalFrames)
                    let smplRatio = Float(frameI + 1) / Float(max(numFrames, 1))
                    updateComparisonProgress(modelBase + modelSpan * (0.70 + 0.25 * smplRatio))
                }
            }

            let smplTime = CFAbsoluteTimeGetCurrent() - smplStart

            let peakMem = max(PerformanceMetrics.currentMemoryMB(), peakMemBefore)
            let totalTime = gvhmrTime + smplTime

            let benchmark = PerformanceMetrics.ModelBenchmark(
                model: model,
                totalTimeSec: totalTime,
                detectTimeSec: detectTime,
                gvhmrTimeSec: gvhmrTime,
                smplTimeSec: smplTime,
                numFrames: numFrames,
                avgGVHMRMs: numFrames > 0 ? (gvhmrTime / Double(numFrames)) * 1000 : 0,
                avgSMPLMs: numFrames > 0 ? (smplTime / Double(numFrames)) * 1000 : 0,
                peakMemoryMB: peakMem
            )

            DispatchQueue.main.async { [weak self] in
                self?.metrics.comparisonResults.append(benchmark)
                self?.comparisonFrameResults[model] = singleResults
                self?.comparisonMultiPersonResults[model] = multiResults
                self?.metrics.comparisonProgress = Float(modelIdx + 1) / Float(max(availableModels.count, 1))
            }
        }

        DispatchQueue.main.async { [weak self] in
            self?.metrics.isComparing = false
            self?.metrics.comparisonProgress = 1.0
            self?.metrics.comparisonPhase = "Done"
            self?.liveActivity.complete(finalPhase: "Comparison done ✓", progress: 1.0)
            self?.endBackgroundTaskIfNeeded()
        }
    }
}
