// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Calvin Chan

import SwiftUI

struct ContentView: View {
    @StateObject private var pipeline = GVHMRPipeline()
    @StateObject private var videoProcessor = VideoProcessor()
    @State private var showInput = true
    @State private var showMesh = true
    @State private var showControls = true
    @State private var statsMode: StatsMode = .all
    @State private var mode: AppMode = .live
    @State private var selectedModel: GVHMRModelChoice = .small
    @State private var showModelPicker = false
    @State private var selectedPreprocessing: PreprocessingMode = .yoloViTPose
    @State private var multiPerson: Bool = false
    @Environment(\.verticalSizeClass) private var verticalSizeClass

    enum AppMode: String, CaseIterable {
        case live = "Live"
        case video = "Video"
    }

    enum StatsMode: String {
        case off = "Stats Off"
        case all = "Stats All"
        case compact = "Stats Core"

        mutating func cycle() {
            switch self {
            case .off: self = .all
            case .all: self = .compact
            case .compact: self = .off
            }
        }
    }

    /// Landscape when verticalSizeClass is compact
    private var isLandscape: Bool { verticalSizeClass == .compact }
    private var statsEnabled: Bool { statsMode != .off }
    private var isVideoProcessingInProgress: Bool {
        let p = videoProcessor.phase
        return p != .idle && p != .done && p != .failed
    }

    var body: some View {
        VStack(spacing: 0) {
            // Top bar: mode picker + status
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("GVHMR")
                        .font(mode == .live ? .subheadline.weight(.bold) : .headline.weight(.bold))
                    if mode == .live {
                        Text(pipeline.status)
                            .font(.caption)
                    }
                }

                Spacer()

                // Model selector button
                Button(action: { showModelPicker.toggle() }) {
                    HStack(spacing: 4) {
                        Image(systemName: "cpu")
                        Text(selectedModel.rawValue)
                    }
                    .font(.caption)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(.ultraThinMaterial)
                    .clipShape(Capsule())
                }

                // Preprocessing toggle — cycles: Vision+ViTPose → YOLO+ViTPose → YOLO-Pose
                Button(action: {
                    let modes = PreprocessingMode.allCases
                    let currentIdx = modes.firstIndex(of: selectedPreprocessing) ?? 0
                    let nextIdx = (currentIdx + 1) % modes.count
                    let newMode = modes[nextIdx]
                    selectedPreprocessing = newMode
                    pipeline.selectPreprocessing(newMode)
                }) {
                    HStack(spacing: 4) {
                        Image(systemName: selectedPreprocessing == .yoloPose ? "bolt.fill" :
                              selectedPreprocessing == .yoloViTPose ? "bolt" : "eye")
                        Text(selectedPreprocessing.rawValue)
                    }
                    .font(.caption)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(selectedPreprocessing == .visionViTPose ? Color.clear :
                                selectedPreprocessing == .yoloViTPose ? Color.green.opacity(0.3) :
                                Color.orange.opacity(0.3))
                    .background(.ultraThinMaterial)
                    .clipShape(Capsule())
                }

                Picker("Mode", selection: $mode) {
                    ForEach(AppMode.allCases, id: \.self) { m in
                        Text(m.rawValue).tag(m)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 140)
            }
            .foregroundColor(.white)
            .padding(.horizontal)
            .padding(.vertical, 6)
            .background(Color.black)

            if isVideoProcessingInProgress {
                Button(action: {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        mode = .video
                    }
                }) {
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("Video: \(videoProcessor.phase.rawValue)")
                                .font(.caption.weight(.semibold))
                                .foregroundColor(.white)
                            Spacer()
                            Text(String(format: "%.0f%%", videoProcessor.progress * 100))
                                .font(.caption2.monospacedDigit())
                                .foregroundColor(.mint)
                            Image(systemName: "chevron.right")
                                .font(.caption2.weight(.semibold))
                                .foregroundColor(.white.opacity(0.7))
                        }

                        ProgressView(value: videoProcessor.progress)
                            .progressViewStyle(.linear)
                            .tint(.mint)
                    }
                    .padding(.horizontal)
                    .padding(.top, 6)
                    .padding(.bottom, 8)
                    .background(Color.black)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }

            // Model picker overlay
            if showModelPicker {
                VStack(spacing: 0) {
                    ForEach(pipeline.availableModels, id: \.self) { model in
                        Button(action: {
                            selectedModel = model
                            pipeline.selectModel(model)
                            showModelPicker = false
                        }) {
                            HStack {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(model.rawValue)
                                        .font(.subheadline.weight(.semibold))
                                    Text(model.detail)
                                        .font(.caption2)
                                        .foregroundColor(.gray)
                                }
                                Spacer()
                                if model == selectedModel {
                                    Image(systemName: "checkmark")
                                        .foregroundColor(.blue)
                                }
                            }
                            .padding(.horizontal, 16)
                            .padding(.vertical, 10)
                            .contentShape(Rectangle())
                        }
                        .foregroundColor(.white)
                        if model != pipeline.availableModels.last {
                            Divider().background(Color.gray.opacity(0.3))
                        }
                    }
                }
                .background(Color(white: 0.15))
                .clipShape(RoundedRectangle(cornerRadius: 10))
                .padding(.horizontal, 16)
                .transition(.opacity)
            }

            // Content area: Live or Video mode
            if mode == .video {
                VideoProcessingView(
                    selectedModel: $selectedModel,
                    selectedPreprocessing: $selectedPreprocessing,
                    processor: videoProcessor
                )
            } else {
                // Adaptive split: HStack in landscape, VStack in portrait
                GeometryReader { geo in
                    let splitView = splitContent(size: geo.size)
                    splitView
                }

                // Bottom controls
                if showControls {
                    controlBar
                }
            }
        }
        .background(Color.black)
        .onAppear {
            pipeline.setup()
            videoProcessor.loadModels()
        }
        .onTapGesture {
            if mode == .live {
                withAnimation { showControls.toggle() }
            }
        }
        .onChange(of: mode) { newMode in
            if newMode == .video {
                pipeline.stop()
            }
        }
        .statusBar(hidden: true)
        .onChange(of: selectedModel) { newModel in
            pipeline.selectModel(newModel)
        }
    }

    // MARK: - Split Content

    @ViewBuilder
    private func splitContent(size: CGSize) -> some View {
        if isLandscape {
            HStack(spacing: 2) {
                cameraPanel
                    .frame(width: size.width / 2)
                posePanel
                    .frame(width: size.width / 2)
            }
        } else {
            VStack(spacing: 2) {
                cameraPanel
                    .frame(height: size.height * 0.55)
                posePanel
                    .frame(height: size.height * 0.45)
            }
        }
    }

    /// Show SMPL mesh (if available) or skeleton
    @ViewBuilder
    private var posePanel: some View {
        ZStack(alignment: .topLeading) {
            if showMesh && !pipeline.smplFaces.isEmpty {
                let liveMeshes: [(vertices: [SIMD3<Float>], trackID: Int, translation: SIMD3<Float>?)]? = {
                    guard multiPerson else { return nil }
                    return pipeline.liveMultiPersonResults.compactMap { p in
                        guard let verts = p.gvhmrResult.meshVertices else { return nil }
                        return (vertices: verts, trackID: p.trackID, translation: p.gvhmrResult.translFullCam)
                    }
                }()
                Mesh3DView(
                    vertices: pipeline.currentResult?.meshVertices,
                    faces: pipeline.smplFaces,
                    multiPersonMeshes: liveMeshes
                )
            } else {
                Pose3DView(result: pipeline.currentResult,
                           debugMessage: pipeline.debugInfo.isEmpty ? nil : pipeline.debugInfo)
            }
        }
    }

    // MARK: - Metrics Overlay

    private var metricsOverlay: some View {
        let m = pipeline.metrics
        return VStack(alignment: .leading, spacing: 3) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("\(selectedModel.rawValue)")
                        .foregroundColor(.cyan)
                    Text(statsMode.rawValue)
                        .foregroundColor(.gray)
                }

                Spacer()

                VStack(alignment: .trailing, spacing: 1) {
                    Text("Inference FPS")
                        .font(.system(size: 10, weight: .semibold))
                        .foregroundColor(.gray)
                    Text(String(format: "%.1f", m.inferenceFPS))
                        .font(.system(size: 14, weight: .bold, design: .monospaced))
                        .foregroundColor(.mint)
                }
            }

            HStack(spacing: 6) {
                metricChip(
                    title: "GVHMR",
                    value: String(format: "%.1f ms", m.gvhmrLatencyMs),
                    tint: .green
                )

                metricChip(
                    title: "Pipeline",
                    value: String(format: "%.1f ms", m.pipelineLatencyMs),
                    tint: .orange
                )
            }

            HStack(spacing: 6) {
                metricChip(
                    title: "Memory",
                    value: String(format: "%.0f MB", m.memoryUsageMB),
                    tint: .blue
                )

                metricChip(
                    title: "Camera",
                    value: String(format: "%.0f FPS", pipeline.fps),
                    tint: .purple
                )
            }

            if statsMode == .all {
                Divider().background(Color.gray.opacity(0.5))
                Text("Preproc: \(selectedPreprocessing.rawValue)")
                    .foregroundColor(.orange)
                Text(String(format: "Caption conf: %.2f", pipeline.captionConfidence))
                Text(String(format: "Detect: %.1f ms", m.poseDetectLatencyMs))
                if selectedPreprocessing == .visionViTPose || selectedPreprocessing == .yoloViTPose {
                    Text(String(format: "ViTPose: %.1f ms", m.vitposeLatencyMs))
                }
                Text(String(format: "SMPL: %.1f ms", m.smplLatencyMs))
                Text("Caption model: \(pipeline.captionModelStatus)")
                    .foregroundColor(.mint)
            }
        }
        .font(.system(size: 11, design: .monospaced))
        .foregroundColor(.white)
        .padding(8)
        .background(Color.black.opacity(0.7))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private func metricChip(title: String, value: String, tint: Color) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(title)
                .font(.system(size: 9, weight: .semibold))
                .foregroundColor(.gray)
            Text(value)
                .font(.system(size: 11, weight: .bold, design: .monospaced))
                .foregroundColor(.white)
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 5)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(tint.opacity(0.2))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    // MARK: - Camera Panel

    private var cameraPanel: some View {
        ZStack {
            CameraPreviewView(session: pipeline.cameraManager.session,
                              isFrontCamera: pipeline.cameraManager.isFrontCamera)

            GeometryReader { geo in
                SkeletonOverlayView(
                    result: pipeline.currentResult,
                    inputKeypoints: showInput ? pipeline.inputKeypoints : [],
                    inputBBox: showInput ? pipeline.inputBBox : .zero,
                    imageSize: pipeline.cameraManager.imageSize,
                    viewSize: geo.size,
                    showInput: showInput,
                    show3D: false,
                    multiPersonResults: multiPerson ? liveMultiPersonOverlay : nil,
                    selectedTrackID: pipeline.selectedTrackID
                )
                .contentShape(Rectangle())
                .onTapGesture { location in
                    guard multiPerson else { return }
                    let selected = selectTrack(at: location, in: geo.size)
                    pipeline.selectPerson(trackID: selected)
                }
            }

            // Camera label + flip button
            VStack {
                HStack {
                    Text(pipeline.cameraManager.isFrontCamera ? "Front" : "Camera")
                        .font(.caption2)
                        .fontWeight(.semibold)
                        .foregroundColor(.white.opacity(0.7))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(.ultraThinMaterial)
                        .clipShape(Capsule())
                    Spacer()
                    Button(action: {
                        pipeline.cameraManager.switchCamera()
                    }) {
                        Image(systemName: "camera.rotate.fill")
                            .font(.system(size: 18))
                            .foregroundColor(.white)
                            .padding(8)
                            .background(.ultraThinMaterial)
                            .clipShape(Circle())
                    }
                }
                .padding(8)

                if pipeline.captionEnabled && pipeline.isRunning {
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Caption")
                                .font(.caption2.weight(.bold))
                                .foregroundColor(.mint)
                            Text("Mode: \(pipeline.captionMode.rawValue)")
                                .font(.caption2)
                                .foregroundColor(.white.opacity(0.85))
                            if multiPerson {
                                Text("Persons: \(pipeline.visiblePersonCount) | Selected: \(pipeline.selectedTrackID.map(String.init) ?? "none")")
                                    .font(.caption2)
                                    .foregroundColor(.white.opacity(0.85))
                            }
                            Text(pipeline.captionText)
                                .font(.caption2)
                                .foregroundColor(.white)
                                .lineLimit(5)
                        }
                        .padding(8)
                        .background(Color.black.opacity(0.65))
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                        Spacer()
                    }
                    .padding(.horizontal, 8)
                }

                Spacer()

                if pipeline.isRunning && statsEnabled {
                    HStack {
                        metricsOverlay
                        Spacer()
                    }
                    .padding(.horizontal, 8)
                    .padding(.bottom, 8)
                }
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    // MARK: - Controls

    private var controlBar: some View {
        VStack(spacing: 8) {
            HStack(spacing: 8) {
                Toggle("2D", isOn: $showInput)
                    .toggleStyle(.button)
                    .tint(showInput ? .cyan : .gray)
                    .font(.caption)
                    .frame(maxWidth: .infinity)

                Toggle("Mesh", isOn: $showMesh)
                    .toggleStyle(.button)
                    .tint(showMesh ? .orange : .gray)
                    .font(.caption)
                    .frame(maxWidth: .infinity)

                Toggle("Multi", isOn: $multiPerson)
                    .toggleStyle(.button)
                    .tint(multiPerson ? .mint : .gray)
                    .font(.caption)
                    .frame(maxWidth: .infinity)
                    .onChange(of: multiPerson) { val in
                        pipeline.isMultiPerson = val
                        if !val {
                            pipeline.liveMultiPersonResults = []
                        }
                    }

                Toggle("Caption", isOn: $pipeline.captionEnabled)
                    .toggleStyle(.button)
                    .tint(pipeline.captionEnabled ? .mint : .gray)
                    .font(.caption)
                    .frame(maxWidth: .infinity)
                    .onChange(of: pipeline.captionEnabled) { enabled in
                        if !enabled {
                            pipeline.captionText = "Semantic stream disabled"
                            pipeline.captionConfidence = 0
                        } else {
                            pipeline.captionText = "Semantic stream warming up"
                        }
                    }
            }

            HStack(spacing: 8) {
                Button(action: { statsMode.cycle() }) {
                    Text(statsMode.rawValue)
                        .font(.caption)
                        .lineLimit(1)
                        .frame(maxWidth: .infinity)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 7)
                        .background(statsMode == .off ? Color.gray.opacity(0.35) : (statsMode == .all ? Color.purple : Color.indigo))
                        .foregroundColor(.white)
                        .clipShape(Capsule())
                }

                Button(action: {
                    pipeline.clearRAM()
                }) {
                    HStack(spacing: 6) {
                        Image(systemName: "trash")
                        Text("Clear RAM")
                    }
                    .font(.caption.weight(.semibold))
                    .lineLimit(1)
                    .frame(maxWidth: .infinity)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 7)
                    .background(Color.orange.opacity(0.85))
                    .foregroundColor(.white)
                    .clipShape(Capsule())
                }

                Button(action: {
                    if pipeline.isRunning {
                        pipeline.stop()
                    } else {
                        pipeline.start()
                    }
                }) {
                    HStack(spacing: 6) {
                        Image(systemName: pipeline.isRunning ? "stop.fill" : "play.fill")
                        Text(pipeline.isRunning ? "Stop" : "Start")
                    }
                    .font(.caption.weight(.semibold))
                    .lineLimit(1)
                    .frame(maxWidth: .infinity)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 7)
                    .background(pipeline.isRunning ? Color.red : Color.green)
                    .foregroundColor(.white)
                    .clipShape(Capsule())
                }
            }
        }
        .frame(minHeight: 50)
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(.ultraThinMaterial)
    }
    /// Convert live multi-person tracked inputs into PersonFrameResult array for 2D overlay.
    private var liveMultiPersonOverlay: [PersonFrameResult]? {
        let inputs = pipeline.multiPersonInputs
        guard !inputs.isEmpty else { return nil }
        let liveMap = Dictionary(uniqueKeysWithValues: pipeline.liveMultiPersonResults.map { ($0.trackID, $0) })
        let emptyResult = GVHMRResult(
            joints3D: [], joints2D: [], predCam: .zero,
            bodyPoseAA: [], globalOrient: .zero,
            betas: [], confidence: 0
        )
        return inputs.enumerated().map { (i, input) in
            if let live = liveMap[input.trackID] {
                return PersonFrameResult(
                    personIndex: i,
                    trackID: input.trackID,
                    keypoints: input.keypoints,
                    bbox: input.bbox,
                    bbxXYS: live.bbxXYS,
                    gvhmrResult: live.gvhmrResult,
                    predX: live.predX
                )
            }
            return PersonFrameResult(
                personIndex: i,
                trackID: input.trackID,
                keypoints: input.keypoints,
                bbox: input.bbox,
                bbxXYS: .zero,
                gvhmrResult: emptyResult,
                predX: []
            )
        }
    }

    private func selectTrack(at location: CGPoint, in viewSize: CGSize) -> Int? {
        let imageSize = pipeline.cameraManager.imageSize
        guard imageSize.width > 1, imageSize.height > 1 else { return nil }

        let scaleX = viewSize.width / imageSize.width
        let scaleY = viewSize.height / imageSize.height
        let scale = max(scaleX, scaleY)
        let offsetX = (viewSize.width - imageSize.width * scale) / 2
        let offsetY = (viewSize.height - imageSize.height * scale) / 2

        let xImg = (location.x - offsetX) / scale
        let yImg = (location.y - offsetY) / scale
        let tapped = CGPoint(x: xImg, y: yImg)

        let tracks = pipeline.multiPersonInputs
        guard !tracks.isEmpty else { return nil }

        if let direct = tracks.first(where: { $0.bbox.contains(tapped) }) {
            return direct.trackID
        }

        var bestID: Int?
        var bestDist = CGFloat.greatestFiniteMagnitude
        for t in tracks {
            let c = CGPoint(x: t.bbox.midX, y: t.bbox.midY)
            let dx = c.x - tapped.x
            let dy = c.y - tapped.y
            let d = sqrt(dx * dx + dy * dy)
            if d < bestDist {
                bestDist = d
                bestID = t.trackID
            }
        }

        return bestDist < 180 ? bestID : nil
    }
}

#Preview {
    ContentView()
}
