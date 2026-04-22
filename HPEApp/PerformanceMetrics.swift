import Foundation
import os

/// Tracks per-stage timing and memory usage for performance comparison across models.
class PerformanceMetrics: ObservableObject {

    // MARK: - Live Mode Metrics (updated in real-time)

    @Published var gvhmrLatencyMs: Double = 0       // GVHMR transformer inference
    @Published var smplLatencyMs: Double = 0         // SMPL forward pass
    @Published var vitposeLatencyMs: Double = 0      // ViTPose keypoint extraction
    @Published var poseDetectLatencyMs: Double = 0   // Apple Vision bbox detection
    @Published var pipelineLatencyMs: Double = 0     // Full pipeline per inference cycle
    @Published var memoryUsageMB: Double = 0         // App memory footprint
    @Published var inferenceFPS: Double = 0          // GVHMR inference rate

    // MARK: - Video Comparison Results

    @Published var comparisonResults: [ModelBenchmark] = []
    @Published var isComparing = false
    @Published var comparisonProgress: Float = 0
    @Published var comparisonPhase: String = ""

    struct ModelBenchmark: Identifiable {
        let id = UUID()
        let model: GVHMRModelChoice
        let totalTimeSec: Double       // Wall-clock total
        let detectTimeSec: Double       // Pose detection phase
        let gvhmrTimeSec: Double        // GVHMR inference phase
        let smplTimeSec: Double         // SMPL mesh phase
        let numFrames: Int
        let avgGVHMRMs: Double          // Average per-window GVHMR latency
        let avgSMPLMs: Double           // Average per-frame SMPL latency
        let peakMemoryMB: Double
    }

    // MARK: - Exponential Moving Average for live metrics

    private let alpha: Double = 0.3  // smoothing factor
    private var rawGVHMR: Double = 0
    private var rawSMPL: Double = 0
    private var rawVitpose: Double = 0
    private var rawPoseDetect: Double = 0
    private var rawPipeline: Double = 0
    private var inferenceCount = 0
    private var lastInferenceCountTime = Date()

    /// Record a GVHMR inference timing.
    func recordGVHMR(_ seconds: Double) {
        rawGVHMR = rawGVHMR == 0 ? seconds : alpha * seconds + (1 - alpha) * rawGVHMR
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.gvhmrLatencyMs = self.rawGVHMR * 1000
        }
        countInference()
    }

    /// Record a SMPL forward pass timing.
    func recordSMPL(_ seconds: Double) {
        rawSMPL = rawSMPL == 0 ? seconds : alpha * seconds + (1 - alpha) * rawSMPL
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.smplLatencyMs = self.rawSMPL * 1000
        }
    }

    /// Record a ViTPose inference timing.
    func recordVitpose(_ seconds: Double) {
        rawVitpose = rawVitpose == 0 ? seconds : alpha * seconds + (1 - alpha) * rawVitpose
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.vitposeLatencyMs = self.rawVitpose * 1000
        }
    }

    /// Record a pose detection timing.
    func recordPoseDetect(_ seconds: Double) {
        rawPoseDetect = rawPoseDetect == 0 ? seconds : alpha * seconds + (1 - alpha) * rawPoseDetect
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.poseDetectLatencyMs = self.rawPoseDetect * 1000
        }
    }

    /// Record full pipeline timing for one inference cycle.
    func recordPipeline(_ seconds: Double) {
        rawPipeline = rawPipeline == 0 ? seconds : alpha * seconds + (1 - alpha) * rawPipeline
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.pipelineLatencyMs = self.rawPipeline * 1000
        }
    }

    private func countInference() {
        inferenceCount += 1
        let now = Date()
        let elapsed = now.timeIntervalSince(lastInferenceCountTime)
        if elapsed >= 1.0 {
            let fps = Double(inferenceCount) / elapsed
            DispatchQueue.main.async { [weak self] in
                self?.inferenceFPS = fps
            }
            inferenceCount = 0
            lastInferenceCountTime = now
        }
    }

    /// Update memory usage (call periodically).
    func updateMemory() {
        let mb = Self.currentMemoryMB()
        DispatchQueue.main.async { [weak self] in
            self?.memoryUsageMB = mb
        }
    }

    /// Reset live metrics (e.g. when switching models).
    func resetLive() {
        rawGVHMR = 0; rawSMPL = 0; rawVitpose = 0; rawPoseDetect = 0; rawPipeline = 0
        inferenceCount = 0; lastInferenceCountTime = Date()
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.gvhmrLatencyMs = 0; self.smplLatencyMs = 0
            self.vitposeLatencyMs = 0; self.poseDetectLatencyMs = 0
            self.pipelineLatencyMs = 0; self.inferenceFPS = 0; self.memoryUsageMB = 0
        }
    }

    // MARK: - Memory Query

    static func currentMemoryMB() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        if kr == KERN_SUCCESS {
            return Double(info.resident_size) / (1024 * 1024)
        }
        return 0
    }

    // MARK: - Timing Helper

    /// Measure a closure's execution time, returning (result, seconds).
    static func measure<T>(_ block: () -> T) -> (T, Double) {
        let start = CFAbsoluteTimeGetCurrent()
        let result = block()
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        return (result, elapsed)
    }
}
