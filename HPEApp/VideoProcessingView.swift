import SwiftUI
import PhotosUI
import Photos
import AVKit

/// Full-screen view for picking a video, processing it offline, and viewing/exporting results.
struct VideoProcessingView: View {
    private struct CompareFullscreenTarget: Identifiable {
        let model: GVHMRModelChoice
        var id: String { model.rawValue }
    }

    private struct PreselectedVideo: Identifiable, Equatable {
        let id = UUID()
        let title: String
        let url: URL
    }

    @Binding var selectedModel: GVHMRModelChoice
    @Binding var selectedPreprocessing: PreprocessingMode
    @ObservedObject var processor: VideoProcessor
    @State private var selectedItems: [PhotosPickerItem] = []
    @State private var preselectedVideos: [PreselectedVideo] = []
    @State private var isImportingPreselectedVideos = false
    @State private var playbackFrame: Int = 0
    @State private var isPlaying = false
    @State private var isExporting = false
    @State private var exportMessage: String?
    @State private var showShareSheet = false
    @State private var shareURLs: [URL] = []
    @State private var showMesh = false
    @State private var frameImage: UIImage?
    @State private var showComparison = false
    @State private var showComparePreviews = false
    @State private var showIncamComparePreview = true
    @State private var multiPersonMode = false
    @State private var videoCaptionEnabled = true
    @State private var selectedVideoTrackID: Int? = nil
    @State private var savedVideos: [SavedProcessedVideo] = []
    @State private var isSavingLibrary = false
    @State private var libraryMessage: String?
    @State private var previewSavedItem: SavedProcessedVideo?
    @State private var fullscreenSavedItem: SavedProcessedVideo?
    @State private var showSavedPreview = false
    @State private var isSavedVideosExpanded = true
    @State private var pendingSavedVideoDeleteID: String?
    @State private var showFullscreenMeshPreview = false
    @State private var fullscreenCompareTarget: CompareFullscreenTarget?
    @State private var hasAutoSavedCurrentProcess = false
    @State private var hasAutoSavedCurrentCompare = false
    @State private var processingStartDate: Date?
    @State private var processingMetricsTick: Int = 0
    @Environment(\.scenePhase) private var scenePhase

    private let timer = Timer.publish(every: 1.0 / 30.0, on: .main, in: .common).autoconnect()

    var body: some View {
        VStack(spacing: 0) {
            headerBar
            Divider()

            if processor.phase == .idle {
                pickVideoSection
            } else if processor.phase == .done {
                resultsView
            } else if processor.phase == .failed {
                failedView
            } else {
                processingView
            }
        }
        .background(Color.black)
        .onChange(of: selectedItems) { newItems in
            importPreselectedItems(newItems)
        }
        .onChange(of: processor.phase) { newPhase in
            if newPhase == .done {
                isPlaying = false
                playbackFrame = 0
                processingStartDate = nil
                loadFrameImage(at: 0)
                if !hasAutoSavedCurrentProcess {
                    hasAutoSavedCurrentProcess = true
                    saveCurrentToLibrary(auto: true)
                }
            } else if newPhase == .idle || newPhase == .failed {
                isPlaying = false
                if newPhase != .failed {
                    processingStartDate = nil
                }
            }
        }
        .onChange(of: processor.metrics.isComparing) { comparing in
            if !comparing,
               !processor.metrics.comparisonResults.isEmpty,
               !hasAutoSavedCurrentCompare {
                hasAutoSavedCurrentCompare = true
                saveCurrentToLibrary(auto: true)
            }
        }
        .sheet(isPresented: $showShareSheet) {
            ShareSheet(urls: shareURLs)
        }
        .sheet(isPresented: $showSavedPreview) {
            if let previewSavedItem {
                SavedVideoPreviewSheet(item: previewSavedItem) {
                    SavedVideoLibrary.shared.delete(previewSavedItem)
                    reloadSavedVideos()
                    showSavedPreview = false
                }
            }
        }
        .fullScreenCover(item: $fullscreenSavedItem) { item in
            SavedVideoFullscreenPlayer(item: item)
        }
        .fullScreenCover(isPresented: $showFullscreenMeshPreview) {
            fullscreenMeshPreviewView
        }
        .fullScreenCover(item: $fullscreenCompareTarget) { target in
            fullscreenCompareMeshPreviewView(for: target.model)
        }
        .onAppear {
            processor.loadModels()
            reloadSavedVideos()
        }
        .onReceive(timer) { _ in
            if processor.phase != .idle && processor.phase != .done && processor.phase != .failed {
                processingMetricsTick = (processingMetricsTick + 1) % 30
                if processingMetricsTick == 0 {
                    processor.metrics.updateMemory()
                }
            }

            if isPlaying {
                if showComparePreviews && !processor.metrics.comparisonResults.isEmpty {
                    let maxFrames = max(
                        processor.comparisonFrameResults[.small]?.count ?? 0,
                        processor.comparisonFrameResults[.medium]?.count ?? 0,
                        processor.comparisonFrameResults[.original]?.count ?? 0
                    )
                    if maxFrames > 0 {
                        playbackFrame = (playbackFrame + 1) % maxFrames
                    }
                } else if processor.phase == .done && !processor.frameResults.isEmpty {
                    playbackFrame = (playbackFrame + 1) % processor.frameResults.count
                }
            }
        }
        .onChange(of: scenePhase) { newPhase in
            processor.handleScenePhase(newPhase)
            if newPhase == .active {
                reloadSavedVideos()
            }
        }
    }

    // MARK: - Header

    private var headerBar: some View {
        HStack {
            Text("GVHMR Video")
                .font(.headline)
                .fontWeight(.bold)

            Text("(\(selectedModel.rawValue))")
                .font(.caption)
                .foregroundColor(.blue)

            if processor.isMultiPerson && processor.personCount > 0 {
                Text("\(processor.personCount)P")
                    .font(.caption2.weight(.bold))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Color.green)
                    .foregroundColor(.white)
                    .clipShape(Capsule())
            }

            Spacer()

            Text(processor.phase.rawValue)
                .font(.caption)
                .foregroundColor(.gray)
        }
        .foregroundColor(.white)
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(Color.black)
    }

    // MARK: - Pick Video

    private var isLimitedLibraryAccess: Bool {
        PHPhotoLibrary.authorizationStatus(for: .readWrite) == .limited
    }

    private var pickVideoSection: some View {
        ScrollView {
            pickVideoMainContent
            .contentShape(Rectangle())
            .onTapGesture {
                pendingSavedVideoDeleteID = nil
            }
            .padding(.vertical, 24)
        }
    }

    @ViewBuilder
    private var pickVideoMainContent: some View {
        VStack(spacing: 24) {
            Image(systemName: "film.stack")
                .font(.system(size: 60))
                .foregroundColor(.blue.opacity(0.6))

            Text("Upload a video for offline GVHMR processing")
                .font(.subheadline)
                .foregroundColor(.gray)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)

            if isImportingPreselectedVideos {
                ProgressView("Importing selected videos...")
                    .tint(.blue)
                    .foregroundColor(.white)
            }

            multiPersonToggle
            addVideosButton

            if isLimitedLibraryAccess {
                manageSelectedPhotosButton
            }

            if !preselectedVideos.isEmpty {
                preselectedVideosSection
            }

            if processor.isReady {
                compareAllModelsPicker
            }

            if showComparison {
                comparisonView
            }

            savedProcessedVideosSection

            if !processor.isReady {
                Text("Models not loaded — check bundle")
                    .font(.caption)
                    .foregroundColor(.red)
            }
        }
    }

    private var multiPersonToggle: some View {
        Toggle(isOn: $multiPersonMode) {
            HStack(spacing: 4) {
                Image(systemName: multiPersonMode ? "person.3.fill" : "person.fill")
                Text(multiPersonMode ? "Multi-Person" : "Single Person")
            }
            .font(.subheadline)
            .foregroundColor(.white)
        }
        .toggleStyle(.button)
        .tint(multiPersonMode ? .green : .gray)
        .padding(.horizontal, 40)
    }

    private var addVideosButton: some View {
        PhotosPicker(selection: $selectedItems, maxSelectionCount: 0, matching: .videos) {
            HStack {
                Image(systemName: "photo.on.rectangle.angled")
                Text(preselectedVideos.isEmpty ? "Choose Videos" : "Add More Videos")
            }
            .font(.headline)
            .padding(.horizontal, 24)
            .padding(.vertical, 12)
            .background(Color.blue)
            .foregroundColor(.white)
            .clipShape(Capsule())
        }
    }

    private var manageSelectedPhotosButton: some View {
        Button(action: { openSelectedPhotosManager() }) {
            HStack(spacing: 8) {
                Image(systemName: "person.crop.rectangle.stack")
                Text("Manage Selected Photos")
            }
            .font(.subheadline.weight(.semibold))
            .padding(.horizontal, 18)
            .padding(.vertical, 10)
            .background(Color.white.opacity(0.14))
            .foregroundColor(.white)
            .clipShape(Capsule())
        }
    }

    private var preselectedVideosSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Preselected Videos")
                    .font(.subheadline.weight(.semibold))
                    .foregroundColor(.white)
                Spacer()
                Text("\(preselectedVideos.count)")
                    .font(.caption2)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(Color.white.opacity(0.15))
                    .foregroundColor(.white)
                    .clipShape(Capsule())
            }

            ForEach(preselectedVideos) { video in
                HStack(spacing: 8) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(video.title)
                            .font(.caption.weight(.semibold))
                            .foregroundColor(.white)
                            .lineLimit(1)
                        Text(video.url.lastPathComponent)
                            .font(.caption2)
                            .foregroundColor(.gray)
                            .lineLimit(1)
                    }
                    Spacer()
                    Button("Process") {
                        processPreselectedVideo(video)
                    }
                    .font(.caption2.weight(.semibold))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(Color.green.opacity(0.85))
                    .foregroundColor(.white)
                    .clipShape(Capsule())

                    Button("Remove") {
                        preselectedVideos.removeAll { $0.id == video.id }
                    }
                    .font(.caption2.weight(.semibold))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(Color.red.opacity(0.85))
                    .foregroundColor(.white)
                    .clipShape(Capsule())
                }
                .padding(8)
                .background(Color.white.opacity(0.08))
                .clipShape(RoundedRectangle(cornerRadius: 8))
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(Color.black.opacity(0.35))
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .padding(.horizontal, 16)
    }

    private var compareAllModelsPicker: some View {
        PhotosPicker(selection: Binding(
            get: { nil },
            set: { item in
                guard let item = item else { return }
                item.loadTransferable(type: VideoTransferable.self) { result in
                    if case .success(let video) = result, let url = video?.url {
                        DispatchQueue.main.async {
                            self.showComparison = true
                            self.processingStartDate = Date()
                            self.hasAutoSavedCurrentCompare = false
                            self.processor.isMultiPerson = self.multiPersonMode
                            self.processor.preprocessingMode = self.selectedPreprocessing
                            self.processor.compareAllModels(url: url)
                        }
                    }
                }
            }
        ), matching: .videos) {
            HStack {
                Image(systemName: "chart.bar.fill")
                Text("Compare All Models")
            }
            .font(.subheadline.weight(.semibold))
            .padding(.horizontal, 20)
            .padding(.vertical, 10)
            .background(Color.orange)
            .foregroundColor(.white)
            .clipShape(Capsule())
        }
    }

    private var savedProcessedVideosSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            savedVideosHeader
            savedVideosExpandedContent
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(Color.black.opacity(0.35))
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .padding(.horizontal, 16)
    }

    private var savedVideosHeader: some View {
        Button(action: {
            pendingSavedVideoDeleteID = nil
            withAnimation(.easeInOut(duration: 0.18)) {
                isSavedVideosExpanded.toggle()
            }
        }) {
            HStack {
                Text("Saved Processed Videos")
                    .font(.subheadline.weight(.semibold))
                    .foregroundColor(.white)
                Spacer()
                Text("\(savedVideos.count)")
                    .font(.caption2)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(Color.white.opacity(0.15))
                    .foregroundColor(.white)
                    .clipShape(Capsule())
                Image(systemName: isSavedVideosExpanded ? "chevron.up" : "chevron.down")
                    .font(.caption.weight(.semibold))
                    .foregroundColor(.white.opacity(0.8))
            }
            .contentShape(Rectangle())
        }
    }

    @ViewBuilder
    private var savedVideosExpandedContent: some View {
        if isSavedVideosExpanded {
            if savedVideos.isEmpty {
                Text("No saved videos yet. Process a video and it will be auto-saved.")
                    .font(.caption)
                    .foregroundColor(.gray)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.vertical, 10)
            } else {
                LazyVStack(spacing: 8) {
                    ForEach(savedVideos) { item in
                        savedVideoRow(item)
                    }
                }
            }
        }
    }

    private func savedVideoRow(_ item: SavedProcessedVideo) -> AnyView {
        let isPendingDelete = pendingSavedVideoDeleteID == item.id
        let deleteIcon = isPendingDelete ? "trash.fill" : "trash"
        let deleteTitle = isPendingDelete ? "Tap Again" : "Delete"
        let deleteHorizontalPadding: CGFloat = isPendingDelete ? 14 : 10
        let deleteBackground: Color = isPendingDelete ? .red : Color.red.opacity(0.75)

        let row = HStack(spacing: 10) {
            VStack(alignment: .leading, spacing: 2) {
                Text(item.title)
                    .font(.caption.weight(.semibold))
                    .foregroundColor(.white)
                    .lineLimit(1)
                Text(savedVideoDate(item.createdAt))
                    .font(.caption2)
                    .foregroundColor(.gray)
            }

            Spacer()

            Button("Preview") {
                pendingSavedVideoDeleteID = nil
                previewSavedItem = item
                showSavedPreview = true
            }
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(Color.blue.opacity(0.75))
            .foregroundColor(.white)
            .clipShape(Capsule())

            Button(action: {
                pendingSavedVideoDeleteID = nil
                fullscreenSavedItem = item
            }) {
                HStack(spacing: 4) {
                    Image(systemName: "arrow.up.left.and.arrow.down.right")
                    Text("Full")
                }
            }
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(Color.purple.opacity(0.75))
            .foregroundColor(.white)
            .clipShape(Capsule())

            Button(action: {
                if isPendingDelete {
                    SavedVideoLibrary.shared.delete(item)
                    pendingSavedVideoDeleteID = nil
                    reloadSavedVideos()
                } else {
                    withAnimation(.easeInOut(duration: 0.15)) {
                        pendingSavedVideoDeleteID = item.id
                    }
                }
            }) {
                HStack(spacing: 4) {
                    Image(systemName: deleteIcon)
                    Text(deleteTitle)
                }
            }
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, deleteHorizontalPadding)
            .padding(.vertical, 6)
            .background(deleteBackground)
            .foregroundColor(.white)
            .clipShape(Capsule())
        }
        .contentShape(Rectangle())
        .onTapGesture {
            if pendingSavedVideoDeleteID != item.id {
                pendingSavedVideoDeleteID = nil
            }
        }
        .padding(8)
        .background(Color.white.opacity(0.08))
        .clipShape(RoundedRectangle(cornerRadius: 8))

        return AnyView(row)
    }

    // MARK: - Processing

    private var processingView: some View {
        ScrollView {
            VStack(spacing: 16) {
                ProgressView(value: processor.progress) {
                    Text(processor.phase.rawValue)
                        .font(.subheadline)
                        .foregroundColor(.white)
                }
                .progressViewStyle(.linear)
                .tint(.blue)
                .padding(.horizontal, 24)

                Text(String(format: "%.0f%%", processor.progress * 100))
                    .font(.title2.weight(.bold))
                    .foregroundColor(.white)

                VStack(alignment: .leading, spacing: 6) {
                    Text("Processing Stats")
                        .font(.caption.weight(.semibold))
                        .foregroundColor(.white.opacity(0.9))

                    Text("Mode: \(multiPersonMode ? "Multi-person" : "Single person")")
                    Text("Model: \(selectedModel.rawValue) | Preproc: \(selectedPreprocessing.rawValue)")
                    Text("Video: \(Int(processor.videoSize.width))x\(Int(processor.videoSize.height)) @ \(String(format: "%.1f", max(processor.fps, 0))) fps")
                    if let start = processingStartDate {
                        Text("Elapsed: \(Int(Date().timeIntervalSince(start)))s")
                    }
                    Text("Pipeline: \(String(format: "%.1f", processor.metrics.pipelineLatencyMs))ms | GVHMR: \(String(format: "%.1f", processor.metrics.gvhmrLatencyMs))ms | SMPL: \(String(format: "%.1f", processor.metrics.smplLatencyMs))ms")
                    Text("Pose: \(String(format: "%.1f", processor.metrics.poseDetectLatencyMs))ms | ViTPose: \(String(format: "%.1f", processor.metrics.vitposeLatencyMs))ms")
                    Text("Memory: \(String(format: "%.0f", processor.metrics.memoryUsageMB)) MB")
                }
                .font(.caption2.monospacedDigit())
                .foregroundColor(.white.opacity(0.85))
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.white.opacity(0.08))
                .clipShape(RoundedRectangle(cornerRadius: 10))
                .padding(.horizontal, 16)

                if !savedVideos.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Recent Saved Videos")
                            .font(.caption.weight(.semibold))
                            .foregroundColor(.white)

                        ForEach(Array(savedVideos.prefix(3))) { item in
                            HStack(spacing: 8) {
                                VStack(alignment: .leading, spacing: 1) {
                                    Text(item.title)
                                        .font(.caption2.weight(.semibold))
                                        .foregroundColor(.white)
                                        .lineLimit(1)
                                    Text(savedVideoDate(item.createdAt))
                                        .font(.caption2)
                                        .foregroundColor(.gray)
                                }
                                Spacer()
                                Button("Preview") {
                                    previewSavedItem = item
                                    showSavedPreview = true
                                }
                                .font(.caption2.weight(.semibold))
                                .padding(.horizontal, 8)
                                .padding(.vertical, 5)
                                .background(Color.blue.opacity(0.75))
                                .foregroundColor(.white)
                                .clipShape(Capsule())
                            }
                            .padding(7)
                            .background(Color.white.opacity(0.08))
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                        }
                    }
                    .padding(.horizontal, 16)
                }

                Button("Cancel") {
                    processor.cancel()
                }
                .foregroundColor(.red)
                .padding(.top, 4)
            }
            .padding(.vertical, 18)
        }
    }

    // MARK: - Failed

    private var failedView: some View {
        VStack(spacing: 16) {
            Spacer()
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 40))
                .foregroundColor(.red)
            Text(processor.error ?? "Processing failed")
                .foregroundColor(.white)
            Button("Try Again") {
                processor.cancel()
            }
            .foregroundColor(.blue)
            Spacer()
        }
    }

    // MARK: - Results

    /// Multi-person results for the current playback frame.
    private var currentMultiPersonFrame: MultiPersonFrameResult? {
        guard processor.isMultiPerson,
              playbackFrame < processor.multiPersonResults.count
        else { return nil }
        return processor.multiPersonResults[playbackFrame]
    }

    /// Semantic caption for the current playback frame.
    private var currentSemanticFrame: CaptionSemanticFrame? {
        guard videoCaptionEnabled else { return nil }

        if processor.isMultiPerson,
           let selected = selectedVideoTrackID,
           let timeline = processor.multiPersonCaptionTimelines[selected] {
            if let exact = timeline.first(where: { $0.frameIndex == playbackFrame }) {
                return exact
            }
            if let nearestPast = timeline.last(where: { $0.frameIndex <= playbackFrame }) {
                return nearestPast
            }
        }

        guard playbackFrame < processor.captionTimeline.count else { return nil }
        return processor.captionTimeline[playbackFrame]
    }

    private var resultsView: some View {
        VStack(spacing: 0) {
            // Playback area
            GeometryReader { geo in
                VStack(spacing: 2) {
                    // Top: Video frame + SMPL incam mesh overlay
                    if playbackFrame < processor.frameResults.count {
                        let result = processor.frameResults[playbackFrame]
                        ZStack {
                            // Video frame background
                            if let img = frameImage {
                                Image(uiImage: img)
                                    .resizable()
                                    .aspectRatio(contentMode: .fit)
                                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                            } else {
                                Rectangle().fill(Color(white: 0.15))
                            }

                            // SMPL mesh overlay (projected incam vertices)
                            Canvas { ctx, size in
                                if let mpFrame = currentMultiPersonFrame, !mpFrame.persons.isEmpty {
                                    drawMultiPersonMeshOnCanvas(ctx: ctx, size: size, persons: mpFrame.persons)
                                } else {
                                    drawMeshOnCanvas(ctx: ctx, size: size, result: result)
                                }
                            }

                            // Frame counter
                            VStack {
                                HStack {
                                    Text("Frame \(playbackFrame + 1) / \(processor.frameResults.count)")
                                        .font(.caption2)
                                        .foregroundColor(.white.opacity(0.7))
                                        .padding(4)
                                        .background(.ultraThinMaterial)
                                        .clipShape(RoundedRectangle(cornerRadius: 4))
                                    if let mpFrame = currentMultiPersonFrame {
                                        Text("\(mpFrame.persons.count) person(s)")
                                            .font(.caption2)
                                            .foregroundColor(.green.opacity(0.8))
                                            .padding(4)
                                            .background(.ultraThinMaterial)
                                            .clipShape(RoundedRectangle(cornerRadius: 4))
                                    }
                                    Spacer()
                                }

                                if processor.isMultiPerson,
                                   let mpFrame = currentMultiPersonFrame,
                                   !mpFrame.persons.isEmpty {
                                    HStack(spacing: 6) {
                                        ForEach(mpFrame.persons.map { $0.trackID }, id: \.self) { trackID in
                                            Button(action: {
                                                selectedVideoTrackID = trackID
                                            }) {
                                                Text("P\(trackID)")
                                                    .font(.caption2.weight(.semibold))
                                                    .padding(.horizontal, 8)
                                                    .padding(.vertical, 4)
                                                    .background(selectedVideoTrackID == trackID ? Color.mint : Color.white.opacity(0.2))
                                                    .foregroundColor(.white)
                                                    .clipShape(Capsule())
                                            }
                                        }
                                        Spacer()
                                    }
                                }

                                    if let semantic = currentSemanticFrame {
                                        HStack {
                                            VStack(alignment: .leading, spacing: 2) {
                                                Text("Caption \(String(format: "%.2f", semantic.confidence))")
                                                    .font(.caption2.weight(.bold))
                                                    .foregroundColor(.mint)
                                                Text(semantic.caption)
                                                    .font(.caption2)
                                                    .foregroundColor(.white)
                                                    .lineLimit(3)
                                            }
                                            .padding(6)
                                            .background(Color.black.opacity(0.65))
                                            .clipShape(RoundedRectangle(cornerRadius: 6))
                                            Spacer()
                                        }
                                    }

                                Spacer()
                            }
                            .padding(8)
                        }
                        .frame(height: geo.size.height * 0.45)
                        .onChange(of: playbackFrame) { _ in
                            loadFrameImage(at: playbackFrame)
                        }
                        .onAppear {
                            loadFrameImage(at: playbackFrame)
                            if selectedVideoTrackID == nil,
                               let first = currentMultiPersonFrame?.persons.first?.trackID {
                                selectedVideoTrackID = first
                            }
                        }

                        // Bottom: 3D global view (mesh or skeleton)
                        if showMesh && !processor.smplFaces.isEmpty {
                            let multiMeshes: [(vertices: [SIMD3<Float>], trackID: Int, translation: SIMD3<Float>?)]? = {
                                guard let mpFrame = currentMultiPersonFrame else { return nil }
                                return mpFrame.persons.compactMap { p in
                                    guard let verts = p.gvhmrResult.meshVertices else { return nil }
                                    return (vertices: verts, trackID: p.trackID, translation: p.gvhmrResult.translFullCam)
                                }
                            }()
                            ZStack(alignment: .topTrailing) {
                                Mesh3DView(
                                    vertices: result.gvhmrResult.meshVertices,
                                    faces: processor.smplFaces,
                                    multiPersonMeshes: multiMeshes
                                )

                                Button(action: {
                                    showFullscreenMeshPreview = true
                                }) {
                                    Image(systemName: "arrow.up.left.and.arrow.down.right")
                                        .font(.caption.weight(.bold))
                                        .padding(8)
                                        .background(Color.black.opacity(0.45))
                                        .clipShape(Circle())
                                }
                                .foregroundColor(.white)
                                .padding(8)
                            }
                            .frame(height: geo.size.height * 0.45)
                        } else {
                            Pose3DView(result: result.gvhmrResult, debugMessage: nil)
                                .frame(height: geo.size.height * 0.45)
                        }
                    }
                }
            }

            if !processor.captionSummary.isEmpty {
                HStack {
                    Text(videoCaptionEnabled ? processor.captionSummary : "Caption disabled")
                        .font(.caption2)
                        .foregroundColor(.mint)
                        .lineLimit(2)
                    Spacer()
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 6)
                .background(Color.black.opacity(0.75))
            }

            // Playback controls
            playbackControls

            // Export controls
            exportControls
        }
    }

    private var playbackControls: some View {
        VStack(spacing: 8) {
            // Scrubber
            Slider(
                value: Binding(
                    get: { Double(playbackFrame) },
                    set: { playbackFrame = Int($0) }
                ),
                in: 0...Double(max(processor.frameResults.count - 1, 1)),
                step: 1
            )
            .tint(.blue)
            .padding(.horizontal)

            HStack(spacing: 16) {
                Button(action: { playbackFrame = max(playbackFrame - 1, 0) }) {
                    Image(systemName: "backward.frame.fill")
                }

                Button(action: { isPlaying.toggle() }) {
                    Image(systemName: isPlaying ? "pause.fill" : "play.fill")
                        .font(.title3)
                }

                Button(action: {
                    playbackFrame = min(playbackFrame + 1, processor.frameResults.count - 1)
                }) {
                    Image(systemName: "forward.frame.fill")
                }

                Spacer()

                Toggle("Mesh", isOn: $showMesh)
                    .toggleStyle(.button)
                    .tint(showMesh ? .orange : .gray)
                    .font(.caption)

                if showMesh {
                    Button(action: {
                        showFullscreenMeshPreview = true
                    }) {
                        Image(systemName: "arrow.up.left.and.arrow.down.right")
                            .font(.caption.weight(.semibold))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 6)
                            .background(Color.white.opacity(0.18))
                            .clipShape(Capsule())
                    }
                    .foregroundColor(.white)
                }

                Toggle("Caption", isOn: $videoCaptionEnabled)
                    .toggleStyle(.button)
                    .tint(videoCaptionEnabled ? .mint : .gray)
                    .font(.caption)
                    .onChange(of: videoCaptionEnabled) { enabled in
                        if enabled,
                           selectedVideoTrackID == nil,
                           let first = currentMultiPersonFrame?.persons.first?.trackID {
                            selectedVideoTrackID = first
                        }
                    }
            }
            .foregroundColor(.white)
            .padding(.horizontal)
        }
        .padding(.vertical, 8)
        .background(.ultraThinMaterial)
    }

    private var exportControls: some View {
        HStack(spacing: 12) {
            Button(action: exportAllOutputs) {
                HStack {
                    Image(systemName: "square.and.arrow.up")
                    Text(isExporting ? "Exporting..." : "Export Videos + Data")
                }
                .font(.subheadline.weight(.semibold))
                .padding(.horizontal, 20)
                .padding(.vertical, 10)
                .background(isExporting ? Color.gray : Color.green)
                .foregroundColor(.white)
                .clipShape(Capsule())
            }
            .disabled(isExporting)

            Button(action: {
                processor.cancel()
                isPlaying = false
                playbackFrame = 0
                frameImage = nil
            }) {
                Text("New Video")
                    .font(.subheadline)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 10)
                    .background(Color.blue.opacity(0.3))
                    .foregroundColor(.white)
                    .clipShape(Capsule())
            }

            if let msg = exportMessage {
                Text(msg)
                    .font(.caption2)
                    .foregroundColor(.green)
            } else if isSavingLibrary {
                Text("Auto-saving...")
                    .font(.caption2)
                    .foregroundColor(.teal)
            } else if let msg = libraryMessage {
                Text(msg)
                    .font(.caption2)
                    .foregroundColor(.mint)
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
        .background(Color.black)
    }

    // MARK: - Drawing Helper

    private func drawSkeletonOnCanvas(ctx: GraphicsContext, size: CGSize, result: GVHMRResult) {
        let scaleX = size.width / processor.videoSize.width
        let scaleY = size.height / processor.videoSize.height
        let scale = min(scaleX, scaleY)
        let offsetX = (size.width - processor.videoSize.width * scale) / 2
        let offsetY = (size.height - processor.videoSize.height * scale) / 2

        func transform(_ pt: CGPoint) -> CGPoint {
            CGPoint(x: pt.x * scale + offsetX, y: pt.y * scale + offsetY)
        }

        // Draw bones
        for (i, bone) in SMPLSkeleton.bones.enumerated() {
            let p1 = transform(result.joints2D[bone.0])
            let p2 = transform(result.joints2D[bone.1])
            let color = SMPLSkeleton.boneColors[i]

            var path = Path()
            path.move(to: p1)
            path.addLine(to: p2)
            ctx.stroke(path, with: .color(Color(
                red: Double(color.0), green: Double(color.1), blue: Double(color.2)
            )), lineWidth: 3)
        }

        // Draw joints
        for pt in result.joints2D {
            let p = transform(pt)
            ctx.fill(Circle().path(in: CGRect(x: p.x - 3, y: p.y - 3, width: 6, height: 6)),
                     with: .color(.white))
        }
    }

    private func drawMeshOnCanvas(ctx: GraphicsContext, size: CGSize, result: VideoProcessor.FrameResult) {
        guard let incamVerts = result.gvhmrResult.meshVerticesIncam,
              let transl = result.gvhmrResult.translFullCam else {
            // Fallback to skeleton if no mesh data
            drawSkeletonOnCanvas(ctx: ctx, size: size, result: result.gvhmrResult)
            return
        }

        let fl = processor.focalLength
        let icx = Float(processor.videoSize.width) / 2
        let icy = Float(processor.videoSize.height) / 2

        let scaleX = Float(size.width) / Float(processor.videoSize.width)
        let scaleY = Float(size.height) / Float(processor.videoSize.height)
        let scale = min(scaleX, scaleY)
        let offsetX = (Float(size.width) - Float(processor.videoSize.width) * scale) / 2
        let offsetY = (Float(size.height) - Float(processor.videoSize.height) * scale) / 2

        // Project incam vertices to 2D and draw as point cloud
        var path = Path()
        for v in incamVerts {
            let vx = v.x + transl.x
            let vy = v.y + transl.y
            let vz = v.z + transl.z
            guard vz > 0.01 else { continue }
            let sx = CGFloat((fl * vx / vz + icx) * scale + offsetX)
            let sy = CGFloat((fl * vy / vz + icy) * scale + offsetY)
            path.addEllipse(in: CGRect(x: sx - 1.5, y: sy - 1.5, width: 3, height: 3))
        }
        ctx.fill(path, with: .color(.white.opacity(0.5)))
    }

    /// Draw multiple colored point-cloud meshes for multi-person mode.
    private func drawMultiPersonMeshOnCanvas(ctx: GraphicsContext, size: CGSize, persons: [PersonFrameResult]) {
        let fl = processor.focalLength
        let icx = Float(processor.videoSize.width) / 2
        let icy = Float(processor.videoSize.height) / 2

        let scaleX = Float(size.width) / Float(processor.videoSize.width)
        let scaleY = Float(size.height) / Float(processor.videoSize.height)
        let scale = min(scaleX, scaleY)
        let offsetX = (Float(size.width) - Float(processor.videoSize.width) * scale) / 2
        let offsetY = (Float(size.height) - Float(processor.videoSize.height) * scale) / 2

        for person in persons {
            guard let incamVerts = person.gvhmrResult.meshVerticesIncam,
                  let transl = person.gvhmrResult.translFullCam else { continue }

            let rgb = PersonColors.color(for: person.trackID)
            let color = Color(red: Double(rgb.0), green: Double(rgb.1), blue: Double(rgb.2))

            var path = Path()
            for v in incamVerts {
                let vx = v.x + transl.x
                let vy = v.y + transl.y
                let vz = v.z + transl.z
                guard vz > 0.01 else { continue }
                let sx = CGFloat((fl * vx / vz + icx) * scale + offsetX)
                let sy = CGFloat((fl * vy / vz + icy) * scale + offsetY)
                path.addEllipse(in: CGRect(x: sx - 1.5, y: sy - 1.5, width: 3, height: 3))
            }
            ctx.fill(path, with: .color(color.opacity(0.5)))
        }
    }

    private struct SmoothedCompareSample {
        let meshVertices: [SIMD3<Float>]?
        let incamVertices: [SIMD3<Float>]?
        let translation: SIMD3<Float>?
    }

    /// Smooth compare preview outputs over a short temporal window to reduce jitter.
    private func smoothedCompareFrame(from result: [VideoProcessor.FrameResult]?, at index: Int, radius: Int = 2) -> SmoothedCompareSample? {
        guard let result, !result.isEmpty else { return nil }
        let clamped = min(max(index, 0), result.count - 1)
        let start = max(0, clamped - radius)
        let end = min(result.count - 1, clamped + radius)
        let window = Array(result[start...end])

        // Average translation where available.
        var t = SIMD3<Float>(repeating: 0)
        var tCount: Float = 0
        for frame in window {
            if let tt = frame.gvhmrResult.translFullCam {
                t += tt
                tCount += 1
            }
        }
        let avgT: SIMD3<Float>? = tCount > 0 ? (t / tCount) : nil

        // Average mesh vertices if all samples have the same topology.
        let meshSamples = window.compactMap { $0.gvhmrResult.meshVertices }
        let avgMesh = averageVertices(meshSamples)

        // Average incam vertices similarly for incam preview mode.
        let incamSamples = window.compactMap { $0.gvhmrResult.meshVerticesIncam }
        let avgIncam = averageVertices(incamSamples)

        return SmoothedCompareSample(meshVertices: avgMesh, incamVertices: avgIncam, translation: avgT)
    }

    private func averageVertices(_ samples: [[SIMD3<Float>]]) -> [SIMD3<Float>]? {
        guard let first = samples.first, !first.isEmpty else { return nil }
        let count = first.count
        guard samples.allSatisfy({ $0.count == count }) else { return first }

        var acc = [SIMD3<Float>](repeating: .zero, count: count)
        for vertices in samples {
            for i in 0..<count {
                acc[i] += vertices[i]
            }
        }
        let n = Float(samples.count)
        for i in 0..<count {
            acc[i] /= n
        }
        return acc
    }

    private func drawIncamComparePreview(
        ctx: GraphicsContext,
        size: CGSize,
        incamVerts: [SIMD3<Float>]?,
        translation: SIMD3<Float>?
    ) {
        guard let incamVerts, let translation else {
            return
        }

        let fl = processor.focalLength
        let icx = Float(processor.videoSize.width) / 2
        let icy = Float(processor.videoSize.height) / 2

        let scaleX = Float(size.width) / Float(max(processor.videoSize.width, 1))
        let scaleY = Float(size.height) / Float(max(processor.videoSize.height, 1))
        let scale = min(scaleX, scaleY)
        let offsetX = (Float(size.width) - Float(processor.videoSize.width) * scale) / 2
        let offsetY = (Float(size.height) - Float(processor.videoSize.height) * scale) / 2

        var path = Path()
        for v in incamVerts {
            let vx = v.x + translation.x
            let vy = v.y + translation.y
            let vz = v.z + translation.z
            guard vz > 0.01 else { continue }
            let sx = CGFloat((fl * vx / vz + icx) * scale + offsetX)
            let sy = CGFloat((fl * vy / vz + icy) * scale + offsetY)
            path.addEllipse(in: CGRect(x: sx - 1.2, y: sy - 1.2, width: 2.4, height: 2.4))
        }
        ctx.fill(path, with: .color(.white.opacity(0.75)))
    }

    private func loadFrameImage(at index: Int) {
        processor.getFrameImage(at: index) { img in
            if let img = img {
                self.frameImage = img
            }
        }
        // Prefetch only when frame images are actually displayed.
        if !showComparePreviews || showIncamComparePreview {
            processor.prefetchFrames(around: index)
        }
    }

    // MARK: - Comparison View

    private var comparisonView: some View {
        let m = processor.metrics
        return VStack(spacing: 12) {
            if m.isComparing {
                VStack(spacing: 8) {
                    Text(m.comparisonPhase)
                        .font(.subheadline)
                        .foregroundColor(.white)
                    ProgressView(value: m.comparisonProgress)
                        .tint(.orange)
                        .padding(.horizontal, 40)
                    Text(String(format: "%.0f%%", m.comparisonProgress * 100))
                        .font(.caption)
                        .foregroundColor(.gray)
                }
            }

            if !m.comparisonResults.isEmpty {
                HStack(spacing: 8) {
                    Button(action: {
                        showComparePreviews.toggle()
                        if showComparePreviews {
                            // Default compare preview mode should start in incam.
                            showIncamComparePreview = true
                            processor.setupFrameExtractor()
                            if let maxCount = processor.comparisonFrameResults.values.map({ $0.count }).max(), maxCount > 0 {
                                playbackFrame = min(playbackFrame, maxCount - 1)
                                loadFrameImage(at: playbackFrame)
                            }
                        }
                    }) {
                        HStack(spacing: 6) {
                            Image(systemName: showComparePreviews ? "rectangle.grid.1x2.fill" : "rectangle.grid.1x2")
                            Text(showComparePreviews ? "Hide 3-Model Preview" : "Show 3-Model Preview")
                        }
                        .font(.caption.weight(.semibold))
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(Color.blue.opacity(0.85))
                        .foregroundColor(.white)
                        .clipShape(Capsule())
                    }

                    if showComparePreviews {
                        Button(action: {
                            showIncamComparePreview.toggle()
                            if showIncamComparePreview {
                                loadFrameImage(at: playbackFrame)
                            }
                        }) {
                            HStack(spacing: 6) {
                                // Button label indicates the next mode to switch to.
                                Image(systemName: showIncamComparePreview ? "cube" : "camera.viewfinder")
                                Text(showIncamComparePreview ? "Mesh" : "Incam")
                            }
                            .font(.caption.weight(.semibold))
                            .padding(.horizontal, 10)
                            .padding(.vertical, 8)
                            .background(Color.white.opacity(0.15))
                            .foregroundColor(.white)
                            .clipShape(Capsule())
                        }
                    }

                    Spacer()
                }
                .padding(.horizontal, 16)
                .padding(.top, 8)

                VStack(spacing: 0) {
                    // Header
                    HStack(spacing: 0) {
                        Text("Model").frame(width: 65, alignment: .leading)
                        Text("GVHMR").frame(width: 55, alignment: .trailing)
                        Text("SMPL").frame(width: 50, alignment: .trailing)
                        Text("Total").frame(width: 50, alignment: .trailing)
                        Text("ms/f").frame(width: 45, alignment: .trailing)
                        Text("MB").frame(width: 45, alignment: .trailing)
                    }
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundColor(.gray)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)

                    Divider().background(Color.gray.opacity(0.5))

                    // Rows
                    ForEach(m.comparisonResults) { result in
                        HStack(spacing: 0) {
                            Text(result.model.rawValue)
                                .frame(width: 65, alignment: .leading)
                                .foregroundColor(.cyan)
                            Text(String(format: "%.1fs", result.gvhmrTimeSec))
                                .frame(width: 55, alignment: .trailing)
                            Text(String(format: "%.1fs", result.smplTimeSec))
                                .frame(width: 50, alignment: .trailing)
                            Text(String(format: "%.1fs", result.totalTimeSec))
                                .frame(width: 50, alignment: .trailing)
                            Text(String(format: "%.0f", result.avgGVHMRMs))
                                .frame(width: 45, alignment: .trailing)
                            Text(String(format: "%.0f", result.peakMemoryMB))
                                .frame(width: 45, alignment: .trailing)
                        }
                        .font(.system(size: 10, design: .monospaced))
                        .foregroundColor(.white)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                    }

                    // Footer with frame count
                    if let first = m.comparisonResults.first {
                        Divider().background(Color.gray.opacity(0.5))
                        Text("\(first.numFrames) frames | Detect: \(String(format: "%.1fs", first.detectTimeSec)) (shared)")
                            .font(.system(size: 9, design: .monospaced))
                            .foregroundColor(.gray)
                            .padding(4)
                    }
                }
                .background(Color(white: 0.1))
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .padding(.horizontal, 16)

                if showComparePreviews {
                    comparePreviewStack
                        .padding(.horizontal, 16)
                        .padding(.top, 10)
                }
            }
        }
        .padding(.bottom, 12)
    }

    private var comparePreviewStack: some View {
        let small = processor.comparisonFrameResults[.small]
        let medium = processor.comparisonFrameResults[.medium]
        let original = processor.comparisonFrameResults[.original]
        let maxFrames = max(small?.count ?? 0, medium?.count ?? 0, original?.count ?? 0)

        return VStack(spacing: 8) {
            if maxFrames > 0 {
                HStack(spacing: 10) {
                    Button(action: { isPlaying.toggle() }) {
                        Image(systemName: isPlaying ? "pause.fill" : "play.fill")
                            .font(.caption)
                            .padding(8)
                            .background(Color.white.opacity(0.15))
                            .clipShape(Circle())
                    }
                    .foregroundColor(.white)

                    Slider(
                        value: Binding(
                            get: { Double(playbackFrame) },
                            set: { playbackFrame = Int($0) }
                        ),
                        in: 0...Double(max(maxFrames - 1, 1)),
                        step: 1
                    )
                    .tint(.blue)
                }
                .onChange(of: playbackFrame) { _ in
                    if showIncamComparePreview {
                        loadFrameImage(at: playbackFrame)
                    }
                }
            }

            comparePreviewPanel(model: .small, title: "Small", result: small, accent: .cyan)
            comparePreviewPanel(model: .medium, title: "Medium", result: medium, accent: .green)
            comparePreviewPanel(model: .original, title: "Original", result: original, accent: .orange)
        }
    }

    @ViewBuilder
    private func comparePreviewPanel(model: GVHMRModelChoice, title: String, result: [VideoProcessor.FrameResult]?, accent: Color) -> some View {
        let smoothed = smoothedCompareFrame(from: result, at: playbackFrame)
        let multiFrame = compareMultiPersons(for: model, at: playbackFrame)

        VStack(spacing: 4) {
            HStack {
                Text(title)
                    .font(.caption.weight(.bold))
                    .foregroundColor(accent)
                Spacer()
                if let result {
                    Text("\(min(playbackFrame + 1, max(result.count, 1)))/\(result.count)")
                        .font(.caption2)
                        .foregroundColor(.gray)
                }
            }

            ZStack {
                if let smoothed {
                    if showIncamComparePreview {
                        if let img = frameImage {
                            Image(uiImage: img)
                                .resizable()
                                .aspectRatio(contentMode: .fit)
                                .frame(maxWidth: .infinity)
                        } else {
                            Rectangle().fill(Color(white: 0.15))
                        }

                        Canvas { ctx, size in
                            if let multiFrame, !multiFrame.isEmpty {
                                drawCompareMultiPersonIncam(ctx: ctx, size: size, persons: multiFrame)
                            } else {
                                drawIncamComparePreview(
                                    ctx: ctx,
                                    size: size,
                                    incamVerts: smoothed.incamVertices,
                                    translation: smoothed.translation
                                )
                            }
                        }
                    } else if !processor.smplFaces.isEmpty {
                        let multiMeshes: [(vertices: [SIMD3<Float>], trackID: Int, translation: SIMD3<Float>?)]? = {
                            guard let multiFrame, !multiFrame.isEmpty else { return nil }
                            return multiFrame.compactMap { person in
                                guard let verts = person.gvhmrResult.meshVertices else { return nil }
                                return (vertices: verts, trackID: person.trackID, translation: person.gvhmrResult.translFullCam)
                            }
                        }()

                        ZStack(alignment: .topTrailing) {
                            Mesh3DView(
                                vertices: multiMeshes == nil ? smoothed.meshVertices : nil,
                                faces: processor.smplFaces,
                                multiPersonMeshes: multiMeshes
                            )

                            Button(action: {
                                fullscreenCompareTarget = CompareFullscreenTarget(model: model)
                            }) {
                                Image(systemName: "arrow.up.left.and.arrow.down.right")
                                    .font(.caption.weight(.bold))
                                    .padding(7)
                                    .background(Color.black.opacity(0.45))
                                    .clipShape(Circle())
                            }
                            .foregroundColor(.white)
                            .padding(6)
                        }
                    } else {
                        Canvas { ctx, size in
                            drawIncamComparePreview(
                                ctx: ctx,
                                size: size,
                                incamVerts: smoothed.incamVertices,
                                translation: smoothed.translation
                            )
                        }
                    }
                } else {
                    Rectangle().fill(Color(white: 0.12))
                    Text("No model output")
                        .font(.caption)
                        .foregroundColor(.gray)
                }
            }
            .frame(height: 150)
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
        .padding(8)
        .background(Color(white: 0.08))
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    private func compareMultiPersons(for model: GVHMRModelChoice, at index: Int) -> [PersonFrameResult]? {
        guard let frames = processor.comparisonMultiPersonResults[model],
              index >= 0,
              index < frames.count else {
            return nil
        }
        return frames[index].persons
    }

    private func drawCompareMultiPersonIncam(ctx: GraphicsContext, size: CGSize, persons: [PersonFrameResult]) {
        let fl = processor.focalLength
        let icx = Float(processor.videoSize.width) / 2
        let icy = Float(processor.videoSize.height) / 2

        let scaleX = Float(size.width) / Float(max(processor.videoSize.width, 1))
        let scaleY = Float(size.height) / Float(max(processor.videoSize.height, 1))
        let scale = min(scaleX, scaleY)
        let offsetX = (Float(size.width) - Float(processor.videoSize.width) * scale) / 2
        let offsetY = (Float(size.height) - Float(processor.videoSize.height) * scale) / 2

        for person in persons {
            guard let incamVerts = person.gvhmrResult.meshVerticesIncam,
                  let transl = person.gvhmrResult.translFullCam else { continue }

            let rgb = PersonColors.color(for: person.trackID)
            let color = Color(red: Double(rgb.0), green: Double(rgb.1), blue: Double(rgb.2))

            var path = Path()
            for v in incamVerts {
                let vx = v.x + transl.x
                let vy = v.y + transl.y
                let vz = v.z + transl.z
                guard vz > 0.01 else { continue }
                let sx = CGFloat((fl * vx / vz + icx) * scale + offsetX)
                let sy = CGFloat((fl * vy / vz + icy) * scale + offsetY)
                path.addEllipse(in: CGRect(x: sx - 1.2, y: sy - 1.2, width: 2.4, height: 2.4))
            }
            ctx.fill(path, with: .color(color.opacity(0.8)))
        }
    }

    private var fullscreenMeshPreviewView: some View {
        ZStack(alignment: .topTrailing) {
            Color.black.ignoresSafeArea()

            if playbackFrame < processor.frameResults.count {
                let frame = processor.frameResults[playbackFrame]
                let multiMeshes: [(vertices: [SIMD3<Float>], trackID: Int, translation: SIMD3<Float>?)]? = {
                    guard let mpFrame = currentMultiPersonFrame else { return nil }
                    return mpFrame.persons.compactMap { p in
                        guard let verts = p.gvhmrResult.meshVertices else { return nil }
                        return (vertices: verts, trackID: p.trackID, translation: p.gvhmrResult.translFullCam)
                    }
                }()

                VStack(spacing: 8) {
                    Mesh3DView(
                        vertices: frame.gvhmrResult.meshVertices,
                        faces: processor.smplFaces,
                        multiPersonMeshes: multiMeshes
                    )
                    .ignoresSafeArea(edges: .bottom)

                    Text("Frame \(playbackFrame + 1) / \(max(processor.frameResults.count, 1))")
                        .font(.caption.monospacedDigit())
                        .foregroundColor(.white.opacity(0.85))
                }
                .padding(.top, 46)
            } else {
                Text("No mesh preview available")
                    .foregroundColor(.gray)
            }

            Button(action: {
                showFullscreenMeshPreview = false
            }) {
                Image(systemName: "xmark.circle.fill")
                    .font(.title2)
                    .foregroundColor(.white)
                    .padding(10)
                    .background(Color.black.opacity(0.45))
                    .clipShape(Circle())
            }
            .padding(.trailing, 14)
            .padding(.top, 10)
        }
    }

    private func fullscreenCompareMeshPreviewView(for model: GVHMRModelChoice) -> some View {
        let frames = processor.comparisonFrameResults[model]
        let smoothed = smoothedCompareFrame(from: frames, at: playbackFrame)
        let multiFrame = compareMultiPersons(for: model, at: playbackFrame)
        let multiMeshes: [(vertices: [SIMD3<Float>], trackID: Int, translation: SIMD3<Float>?)]? = {
            guard let multiFrame, !multiFrame.isEmpty else { return nil }
            return multiFrame.compactMap { person in
                guard let verts = person.gvhmrResult.meshVertices else { return nil }
                return (vertices: verts, trackID: person.trackID, translation: person.gvhmrResult.translFullCam)
            }
        }()

        return ZStack(alignment: .topTrailing) {
            Color.black.ignoresSafeArea()

            VStack(spacing: 8) {
                Mesh3DView(
                    vertices: multiMeshes == nil ? smoothed?.meshVertices : nil,
                    faces: processor.smplFaces,
                    multiPersonMeshes: multiMeshes
                )
                .ignoresSafeArea(edges: .bottom)

                let total = frames?.count ?? 0
                Text("\(model.rawValue)  •  Frame \(min(playbackFrame + 1, max(total, 1))) / \(max(total, 1))")
                    .font(.caption.monospacedDigit())
                    .foregroundColor(.white.opacity(0.85))
            }
            .padding(.top, 46)

            Button(action: {
                fullscreenCompareTarget = nil
            }) {
                Image(systemName: "xmark.circle.fill")
                    .font(.title2)
                    .foregroundColor(.white)
                    .padding(10)
                    .background(Color.black.opacity(0.45))
                    .clipShape(Circle())
            }
            .padding(.trailing, 14)
            .padding(.top, 10)
        }
    }

    // MARK: - Export

    private func exportAllOutputs() {
        isExporting = true
        exportMessage = nil

        DispatchQueue.global(qos: .userInitiated).async {
            let dir = FileManager.default.temporaryDirectory
                .appendingPathComponent("gvhmr_export_\(Int(Date().timeIntervalSince1970))")
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

            let incamURL = dir.appendingPathComponent("1_incam.mp4")
            let globalURL = dir.appendingPathComponent("2_global.mp4")
            let jsonURL = dir.appendingPathComponent("hmr4d_results.json")

            let group = DispatchGroup()
            let urlQueue = DispatchQueue(label: "com.gvhmr.exportURLs")
            var urls = [URL]()

            // Export incam
            if let srcURL = self.processor.sourceVideoURL {
                group.enter()
                if self.processor.isMultiPerson && !self.processor.multiPersonResults.isEmpty {
                    VideoExporter.exportIncamVideoMulti(
                        sourceURL: srcURL,
                        multiResults: self.processor.multiPersonResults,
                        smplFaces: self.processor.smplFaces,
                        focalLength: self.processor.focalLength,
                        outputURL: incamURL
                    ) { success, msg in
                        if success { urlQueue.sync { urls.append(incamURL) } }
                        print("[Export] incam multi: \(msg)")
                        group.leave()
                    }
                } else {
                    VideoExporter.exportIncamVideo(
                        sourceURL: srcURL,
                        results: self.processor.frameResults,
                        smplFaces: self.processor.smplFaces,
                        focalLength: self.processor.focalLength,
                        outputURL: incamURL
                    ) { success, msg in
                        if success { urlQueue.sync { urls.append(incamURL) } }
                        print("[Export] incam: \(msg)")
                        group.leave()
                    }
                }
            }

            // Export global
            group.enter()
            if self.processor.isMultiPerson && !self.processor.multiPersonResults.isEmpty {
                VideoExporter.exportGlobalVideoMulti(
                    multiResults: self.processor.multiPersonResults,
                    smplFaces: self.processor.smplFaces,
                    videoSize: self.processor.videoSize,
                    fps: self.processor.fps,
                    outputURL: globalURL
                ) { success, msg in
                    if success { urlQueue.sync { urls.append(globalURL) } }
                    print("[Export] global multi: \(msg)")
                    group.leave()
                }
            } else {
                VideoExporter.exportGlobalVideo(
                    results: self.processor.frameResults,
                    smplFaces: self.processor.smplFaces,
                    videoSize: self.processor.videoSize,
                    fps: self.processor.fps,
                    outputURL: globalURL
                ) { success, msg in
                    if success { urlQueue.sync { urls.append(globalURL) } }
                    print("[Export] global: \(msg)")
                    group.leave()
                }
            }

            // Export JSON
            let jsonExported: Bool
            if self.processor.isMultiPerson && !self.processor.multiPersonResults.isEmpty {
                jsonExported = VideoExporter.exportResultsMulti(
                    multiResults: self.processor.multiPersonResults,
                    videoSize: self.processor.videoSize,
                    fps: self.processor.fps,
                    outputURL: jsonURL,
                    captionTimeline: self.processor.captionTimeline
                )
            } else {
                jsonExported = VideoExporter.exportResults(
                    results: self.processor.frameResults,
                    videoSize: self.processor.videoSize,
                    fps: self.processor.fps,
                    outputURL: jsonURL,
                    captionTimeline: self.processor.captionTimeline
                )
            }
            if jsonExported {
                urls.append(jsonURL)
            }

            group.wait()

            let finalURLs = urlQueue.sync { urls }
            DispatchQueue.main.async {
                self.isExporting = false
                self.shareURLs = finalURLs
                self.exportMessage = "\(finalURLs.count) files exported"
                if !finalURLs.isEmpty {
                    self.showShareSheet = true
                }
            }
        }
    }

    private func saveCurrentToLibrary(auto: Bool = false) {
        let isCompareSession = !processor.metrics.comparisonResults.isEmpty
        guard let srcURL = processor.sourceVideoURL else {
            return
        }

        isSavingLibrary = true
        libraryMessage = nil

        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let benchData: Data? = {
                    guard !self.processor.metrics.comparisonResults.isEmpty else { return nil }
                    let benchObjects: [[String: Any]] = self.processor.metrics.comparisonResults.map { b in
                        [
                            "model": b.model.rawValue,
                            "total_time_sec": b.totalTimeSec,
                            "detect_time_sec": b.detectTimeSec,
                            "gvhmr_time_sec": b.gvhmrTimeSec,
                            "smpl_time_sec": b.smplTimeSec,
                            "num_frames": b.numFrames,
                            "avg_gvhmr_ms": b.avgGVHMRMs,
                            "avg_smpl_ms": b.avgSMPLMs,
                            "peak_memory_mb": b.peakMemoryMB,
                        ]
                    }
                    return try? JSONSerialization.data(withJSONObject: benchObjects, options: .prettyPrinted)
                }()

                func exportEntry(
                    title: String,
                    frameResults: [VideoProcessor.FrameResult],
                    multiResults: [MultiPersonFrameResult],
                    sessionType: String,
                    compareCompositeFileName: String? = nil
                ) throws {
                    let entry = try SavedVideoLibrary.shared.makeEntryDirectory()
                    let folderName = entry.folderName
                    let dir = entry.folderURL

                    let incamPreviewURL = dir.appendingPathComponent("1_incam.mp4")
                    let meshPreviewURL = dir.appendingPathComponent("2_global.mp4")
                    let jsonURL = dir.appendingPathComponent("hmr4d_results.json")
                    let benchmarkURL = dir.appendingPathComponent("compare_benchmarks.json")

                    var hasIncam = false
                    if !multiResults.isEmpty {
                        let sem = DispatchSemaphore(value: 0)
                        VideoExporter.exportIncamVideoMulti(
                            sourceURL: srcURL,
                            multiResults: multiResults,
                            smplFaces: self.processor.smplFaces,
                            focalLength: self.processor.focalLength,
                            outputURL: incamPreviewURL
                        ) { success, _ in
                            hasIncam = success
                            sem.signal()
                        }
                        sem.wait()
                    } else if !frameResults.isEmpty {
                        let sem = DispatchSemaphore(value: 0)
                        VideoExporter.exportIncamVideo(
                            sourceURL: srcURL,
                            results: frameResults,
                            smplFaces: self.processor.smplFaces,
                            focalLength: self.processor.focalLength,
                            outputURL: incamPreviewURL
                        ) { success, _ in
                            hasIncam = success
                            sem.signal()
                        }
                        sem.wait()
                    }

                    var hasMeshPreview = false
                    if !multiResults.isEmpty {
                        let sem = DispatchSemaphore(value: 0)
                        VideoExporter.exportGlobalVideoMulti(
                            multiResults: multiResults,
                            smplFaces: self.processor.smplFaces,
                            videoSize: self.processor.videoSize,
                            fps: self.processor.fps,
                            outputURL: meshPreviewURL
                        ) { success, _ in
                            hasMeshPreview = success
                            sem.signal()
                        }
                        sem.wait()
                    } else if !frameResults.isEmpty {
                        let sem = DispatchSemaphore(value: 0)
                        VideoExporter.exportGlobalVideo(
                            results: frameResults,
                            smplFaces: self.processor.smplFaces,
                            videoSize: self.processor.videoSize,
                            fps: self.processor.fps,
                            outputURL: meshPreviewURL
                        ) { success, _ in
                            hasMeshPreview = success
                            sem.signal()
                        }
                        sem.wait()
                    }

                    let hasJSON: Bool
                    if !multiResults.isEmpty {
                        hasJSON = VideoExporter.exportResultsMulti(
                            multiResults: multiResults,
                            videoSize: self.processor.videoSize,
                            fps: self.processor.fps,
                            outputURL: jsonURL,
                            captionTimeline: self.processor.captionTimeline
                        )
                    } else {
                        hasJSON = VideoExporter.exportResults(
                            results: frameResults,
                            videoSize: self.processor.videoSize,
                            fps: self.processor.fps,
                            outputURL: jsonURL,
                            captionTimeline: self.processor.captionTimeline
                        )
                    }

                    var hasBenchmark = false
                    if let benchData {
                        hasBenchmark = FileManager.default.createFile(atPath: benchmarkURL.path, contents: benchData)
                    }

                    let item = SavedProcessedVideo(
                        title: title,
                        sourceVideoName: srcURL.lastPathComponent,
                        folderName: folderName,
                        incamFileName: hasIncam ? "1_incam.mp4" : nil,
                        globalFileName: hasMeshPreview ? "2_global.mp4" : nil,
                        jsonFileName: hasJSON ? "hmr4d_results.json" : nil,
                        benchmarkFileName: hasBenchmark ? "compare_benchmarks.json" : nil,
                        compareCompositeFileName: compareCompositeFileName,
                        sessionType: sessionType
                    )

                    if item.incamFileName == nil && item.globalFileName == nil && item.compareCompositeFileName == nil {
                        throw NSError(
                            domain: "SavedVideo",
                            code: -1,
                            userInfo: [NSLocalizedDescriptionKey: "No preview outputs were generated"]
                        )
                    }

                    SavedVideoLibrary.shared.append(item)
                }

                let titleBase = srcURL.deletingPathExtension().lastPathComponent

                if isCompareSession {
                    let entry = try SavedVideoLibrary.shared.makeEntryDirectory()
                    let folderName = entry.folderName
                    let dir = entry.folderURL
                    let compareVideoName = "compare_labeled.mp4"
                    let compareVideoURL = dir.appendingPathComponent(compareVideoName)
                    let compareSMPLName = "compare_smpl.mp4"
                    let compareSMPLURL = dir.appendingPathComponent(compareSMPLName)
                    let benchmarkURL = dir.appendingPathComponent("compare_benchmarks.json")

                    var hasCompareLabeled = false
                    var hasCompareSMPL = false

                    let semLabeled = DispatchSemaphore(value: 0)
                    VideoExporter.exportComparisonLabeledVideo(
                        sourceURL: srcURL,
                        comparisonFrameResults: self.processor.comparisonFrameResults,
                        comparisonMultiPersonResults: self.processor.comparisonMultiPersonResults,
                        fps: self.processor.fps,
                        outputURL: compareVideoURL
                    ) { success, _ in
                        hasCompareLabeled = success
                        semLabeled.signal()
                    }
                    semLabeled.wait()

                    let semSMPL = DispatchSemaphore(value: 0)
                    VideoExporter.exportComparisonCompositeVideo(
                        sourceURL: srcURL,
                        comparisonFrameResults: self.processor.comparisonFrameResults,
                        comparisonMultiPersonResults: self.processor.comparisonMultiPersonResults,
                        benchmarks: self.processor.metrics.comparisonResults,
                        smplFaces: self.processor.smplFaces,
                        fps: self.processor.fps,
                        outputURL: compareSMPLURL
                    ) { success, _ in
                        hasCompareSMPL = success
                        semSMPL.signal()
                    }
                    semSMPL.wait()

                    var hasBenchmark = false
                    if let benchData {
                        hasBenchmark = FileManager.default.createFile(atPath: benchmarkURL.path, contents: benchData)
                    }

                    guard hasCompareLabeled || hasCompareSMPL else {
                        throw NSError(
                            domain: "SavedVideo",
                            code: -2,
                            userInfo: [NSLocalizedDescriptionKey: "No compare outputs were available to save"]
                        )
                    }

                    let compareItem = SavedProcessedVideo(
                        title: "\(titleBase) [Compare All Models]",
                        sourceVideoName: srcURL.lastPathComponent,
                        folderName: folderName,
                        incamFileName: hasCompareLabeled ? compareVideoName : nil,
                        globalFileName: hasCompareSMPL ? compareSMPLName : nil,
                        jsonFileName: nil,
                        benchmarkFileName: hasBenchmark ? "compare_benchmarks.json" : nil,
                        compareCompositeFileName: hasCompareLabeled ? compareVideoName : (hasCompareSMPL ? compareSMPLName : nil),
                        sessionType: "compare"
                    )
                    SavedVideoLibrary.shared.append(compareItem)
                } else {
                    try exportEntry(
                        title: titleBase,
                        frameResults: self.processor.frameResults,
                        multiResults: self.processor.multiPersonResults,
                        sessionType: "process"
                    )
                }

                DispatchQueue.main.async {
                    self.isSavingLibrary = false
                    self.libraryMessage = isCompareSession ? (auto ? "Auto-saved compare models" : "Saved compare models") : (auto ? "Auto-saved" : "Saved")
                    self.reloadSavedVideos()
                }
            } catch {
                DispatchQueue.main.async {
                    self.isSavingLibrary = false
                    self.libraryMessage = "Save failed"
                    print("[SavedVideo] Save failed: \(error)")
                }
            }
        }
    }

    private func reloadSavedVideos() {
        savedVideos = SavedVideoLibrary.shared.loadItems()
    }

    private func savedVideoDate(_ date: Date) -> String {
        let f = DateFormatter()
        f.dateStyle = .medium
        f.timeStyle = .short
        return f.string(from: date)
    }

    // MARK: - Video Loading

    private func importPreselectedItems(_ items: [PhotosPickerItem]) {
        guard !items.isEmpty else { return }

        isImportingPreselectedVideos = true
        let group = DispatchGroup()
        let lock = DispatchQueue(label: "com.gvhmr.preselected.lock")
        var importedVideos: [PreselectedVideo] = []

        for item in items {
            group.enter()
            item.loadTransferable(type: VideoTransferable.self) { result in
                defer { group.leave() }

                if case .success(let video) = result,
                   let url = video?.url {
                    let title = url.deletingPathExtension().lastPathComponent
                    let preselected = PreselectedVideo(title: title, url: url)
                    lock.sync {
                        importedVideos.append(preselected)
                    }
                }
            }
        }

        group.notify(queue: .main) {
            let existingPaths = Set(preselectedVideos.map { $0.url.path })
            let filtered = importedVideos.filter { !existingPaths.contains($0.url.path) }
            preselectedVideos.append(contentsOf: filtered)
            selectedItems = []
            isImportingPreselectedVideos = false
        }
    }

    private func processPreselectedVideo(_ video: PreselectedVideo) {
        processSelectedVideoURL(video.url)
    }

    private func processSelectedVideoURL(_ url: URL) {
        processingStartDate = Date()
        hasAutoSavedCurrentProcess = false
        hasAutoSavedCurrentCompare = false
        showComparePreviews = false
        showIncamComparePreview = true
        isPlaying = false
        playbackFrame = 0
        frameImage = nil
        selectedVideoTrackID = nil
        processor.isMultiPerson = multiPersonMode

        // Apply selected model and preprocessing mode before processing
        processor.selectModel(selectedModel)
        processor.preprocessingMode = selectedPreprocessing

        isPlaying = false
        playbackFrame = 0
        frameImage = nil
        selectedVideoTrackID = nil

        if multiPersonMode {
            processor.processVideoMulti(url: url)
        } else {
            processor.processVideo(url: url)
        }
    }

    private func openSelectedPhotosManager() {
        let status = PHPhotoLibrary.authorizationStatus(for: .readWrite)
        if status == .notDetermined {
            PHPhotoLibrary.requestAuthorization(for: .readWrite) { newStatus in
                if newStatus == .limited {
                    DispatchQueue.main.async {
                        openSelectedPhotosManager()
                    }
                }
            }
            return
        }

        guard status == .limited else {
            libraryMessage = "Set Photos access to Selected Photos in iOS Settings"
            return
        }

        guard let scene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
              let rootVC = scene.windows.first?.rootViewController else {
            return
        }

        PHPhotoLibrary.shared().presentLimitedLibraryPicker(from: rootVC)
    }
}

// MARK: - Video Transferable

struct VideoTransferable: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { video in
            SentTransferredFile(video.url)
        } importing: { received in
            let destDir = FileManager.default.temporaryDirectory
                .appendingPathComponent("gvhmr_input")
            try? FileManager.default.createDirectory(at: destDir, withIntermediateDirectories: true)
            let dest = destDir.appendingPathComponent(received.file.lastPathComponent)
            if FileManager.default.fileExists(atPath: dest.path) {
                try? FileManager.default.removeItem(at: dest)
            }
            try FileManager.default.copyItem(at: received.file, to: dest)
            return VideoTransferable(url: dest)
        }
    }
}

// MARK: - Share Sheet

struct ShareSheet: UIViewControllerRepresentable {
    let urls: [URL]

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: urls, applicationActivities: nil)
    }

    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}

// MARK: - Saved Preview Sheet

private struct SavedVideoPreviewSheet: View {
    private struct BenchmarkRow: Identifiable {
        let id = UUID()
        let model: String
        let gvhmrSec: Double
        let smplSec: Double
        let totalSec: Double
        let avgGVHMRMs: Double
        let memoryMB: Double
        let detectSec: Double
        let numFrames: Int
    }

    let item: SavedProcessedVideo
    let onDelete: () -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var selectedTab = 0
    @State private var player = AVPlayer()
    @State private var captionSummaryText: String = ""
    @State private var benchmarkRows: [BenchmarkRow] = []

    private var compareURL: URL? { SavedVideoLibrary.shared.resolveCompareCompositeURL(for: item) }
    private var incamURL: URL? { SavedVideoLibrary.shared.resolveIncamURL(for: item) }
    private var globalURL: URL? { SavedVideoLibrary.shared.resolveGlobalURL(for: item) }
    private var isCompareSession: Bool { item.sessionType == "compare" }
    private var currentURL: URL? {
        if isCompareSession {
            if selectedTab == 0 {
                return compareURL ?? incamURL ?? globalURL
            }
            return globalURL ?? compareURL ?? incamURL
        }
        return selectedTab == 0 ? (incamURL ?? globalURL) : (globalURL ?? incamURL)
    }

    var body: some View {
        NavigationView {
            VStack(spacing: 10) {
                Picker("Video", selection: $selectedTab) {
                    if isCompareSession {
                        Text("Compare").tag(0)
                        Text("SMPL").tag(1)
                    } else {
                        Text("Incam").tag(0)
                        Text("Global").tag(1)
                    }
                }
                .pickerStyle(.segmented)
                .padding(.horizontal)

                if currentURL != nil {
                    VideoPlayer(player: player)
                        .frame(height: 280)
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                        .padding(.horizontal)
                } else {
                    RoundedRectangle(cornerRadius: 10)
                        .fill(Color.gray.opacity(0.2))
                        .frame(height: 280)
                        .overlay(Text("No preview video found").foregroundColor(.secondary))
                        .padding(.horizontal)
                }

                if !captionSummaryText.isEmpty {
                    Text(captionSummaryText)
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal)
                }

                if !benchmarkRows.isEmpty {
                    VStack(alignment: .leading, spacing: 6) {
                        Text((item.sessionType == "compare") ? "Compare Benchmarks" : "Benchmarks")
                            .font(.subheadline.weight(.semibold))
                            .padding(.horizontal)

                        ScrollView(.horizontal, showsIndicators: false) {
                            VStack(spacing: 0) {
                                HStack(spacing: 0) {
                                    benchmarkHeaderCell("Model", width: 65, align: .leading)
                                    benchmarkHeaderCell("GVHMR", width: 55, align: .trailing)
                                    benchmarkHeaderCell("SMPL", width: 50, align: .trailing)
                                    benchmarkHeaderCell("Total", width: 50, align: .trailing)
                                    benchmarkHeaderCell("ms/f", width: 45, align: .trailing)
                                    benchmarkHeaderCell("MB", width: 45, align: .trailing)
                                }

                                Divider().background(Color.gray.opacity(0.4))

                                ForEach(benchmarkRows) { row in
                                    HStack(spacing: 0) {
                                        benchmarkValueCell(row.model, width: 65, strong: true, align: .leading)
                                        benchmarkValueCell(String(format: "%.1fs", row.gvhmrSec), width: 55, align: .trailing)
                                        benchmarkValueCell(String(format: "%.1fs", row.smplSec), width: 50, align: .trailing)
                                        benchmarkValueCell(String(format: "%.1fs", row.totalSec), width: 50, align: .trailing)
                                        benchmarkValueCell(String(format: "%.0f", row.avgGVHMRMs), width: 45, align: .trailing)
                                        benchmarkValueCell(String(format: "%.0f", row.memoryMB), width: 45, align: .trailing)
                                    }
                                }

                                if let first = benchmarkRows.first {
                                    Divider().background(Color.gray.opacity(0.4))
                                    Text("\(first.numFrames) frames | Detect: \(String(format: "%.1fs", first.detectSec)) (shared)")
                                        .font(.system(size: 9, design: .monospaced))
                                        .foregroundColor(.gray)
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                        .padding(.horizontal, 8)
                                        .padding(.vertical, 4)
                                }
                            }
                            .overlay(
                                RoundedRectangle(cornerRadius: 8)
                                    .stroke(Color.white.opacity(0.2), lineWidth: 1)
                            )
                            .padding(.horizontal)
                        }
                    }
                    .frame(maxHeight: 180)
                }

                Spacer()
            }
            .navigationTitle(item.title)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Close") { dismiss() }
                }
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(role: .destructive) {
                        player.pause()
                        onDelete()
                    } label: {
                        Label("Delete", systemImage: "trash")
                    }
                }
            }
            .onAppear {
                loadMeta()
                playCurrent()
            }
            .onChange(of: selectedTab) { _ in
                playCurrent()
            }
            .onDisappear {
                player.pause()
            }
        }
    }

    private func playCurrent() {
        guard let url = currentURL else { return }
        player.replaceCurrentItem(with: AVPlayerItem(url: url))
        player.play()
    }

    private func loadMeta() {
        captionSummaryText = ""
        benchmarkRows = []

        if let jsonURL = SavedVideoLibrary.shared.resolveJSONURL(for: item),
           let data = try? Data(contentsOf: jsonURL),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let frames = obj["frames"] as? [[String: Any]] {
            let captionCount = frames.compactMap { $0["caption"] as? [String: Any] }.count
            if captionCount > 0 {
                captionSummaryText = "Captions: \(captionCount) frames"
            }
        }

        if let benchURL = SavedVideoLibrary.shared.resolveBenchmarkURL(for: item),
           let data = try? Data(contentsOf: benchURL),
           let obj = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
            benchmarkRows = obj.map { row in
                BenchmarkRow(
                    model: "\(row["model"] ?? "-")",
                    gvhmrSec: row["gvhmr_time_sec"] as? Double ?? 0,
                    smplSec: row["smpl_time_sec"] as? Double ?? 0,
                    totalSec: row["total_time_sec"] as? Double ?? 0,
                    avgGVHMRMs: row["avg_gvhmr_ms"] as? Double ?? 0,
                    memoryMB: row["peak_memory_mb"] as? Double ?? 0,
                    detectSec: row["detect_time_sec"] as? Double ?? 0,
                    numFrames: row["num_frames"] as? Int ?? 0
                )
            }
        }
    }

    private func benchmarkHeaderCell(_ text: String, width: CGFloat, align: Alignment) -> some View {
        Text(text)
            .font(.system(size: 10, weight: .bold, design: .monospaced))
            .foregroundColor(.gray)
            .frame(width: width, alignment: align)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(Color.black.opacity(0.2))
    }

    private func benchmarkValueCell(_ text: String, width: CGFloat, strong: Bool = false, align: Alignment) -> some View {
        Text(text)
            .font(.system(size: 10, weight: strong ? .semibold : .regular, design: .monospaced))
            .foregroundColor(.white.opacity(strong ? 1.0 : 0.9))
            .frame(width: width, alignment: align)
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(Color.black.opacity(0.2))
    }
}

private struct SavedVideoFullscreenPlayer: View {
    let item: SavedProcessedVideo

    @Environment(\.dismiss) private var dismiss
    @State private var selectedTab = 0
    @State private var player = AVPlayer()

    private var compareURL: URL? { SavedVideoLibrary.shared.resolveCompareCompositeURL(for: item) }
    private var incamURL: URL? { SavedVideoLibrary.shared.resolveIncamURL(for: item) }
    private var globalURL: URL? { SavedVideoLibrary.shared.resolveGlobalURL(for: item) }
    private var isCompareSession: Bool { item.sessionType == "compare" }

    private var currentURL: URL? {
        if isCompareSession {
            if selectedTab == 0 {
                return compareURL ?? incamURL ?? globalURL
            }
            return globalURL ?? compareURL ?? incamURL
        }
        return selectedTab == 0 ? (incamURL ?? globalURL) : (globalURL ?? incamURL)
    }

    var body: some View {
        ZStack(alignment: .topTrailing) {
            Color.black.ignoresSafeArea()

            VStack(spacing: 10) {
                Picker("Video", selection: $selectedTab) {
                    if isCompareSession {
                        Text("Compare").tag(0)
                        Text("SMPL").tag(1)
                    } else {
                        Text("Incam").tag(0)
                        Text("Global").tag(1)
                    }
                }
                .pickerStyle(.segmented)
                .padding(.horizontal, 16)
                .padding(.top, 48)

                if currentURL != nil {
                    VideoPlayer(player: player)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    Spacer()
                    Text("No preview video found")
                        .foregroundColor(.gray)
                    Spacer()
                }
            }

            Button(action: { dismiss() }) {
                Image(systemName: "xmark.circle.fill")
                    .font(.title2)
                    .foregroundColor(.white)
                    .padding(10)
                    .background(Color.black.opacity(0.45))
                    .clipShape(Circle())
            }
            .padding(.trailing, 14)
            .padding(.top, 10)
        }
        .onAppear {
            playCurrent()
        }
        .onChange(of: selectedTab) { _ in
            playCurrent()
        }
        .onDisappear {
            player.pause()
        }
    }

    private func playCurrent() {
        guard let url = currentURL else {
            player.pause()
            return
        }
        player.replaceCurrentItem(with: AVPlayerItem(url: url))
        player.play()
    }
}
