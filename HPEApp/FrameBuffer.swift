import Foundation

/// Circular buffer that accumulates per-frame data for the GVHMR temporal window.
class FrameBuffer {
    let capacity: Int
    private(set) var frames: [FrameData] = []
    private var frameCount = 0

    init(capacity: Int = GVHMRConstants.windowSize) {
        self.capacity = capacity
    }

    /// Add a new frame. Drops oldest frame when full.
    func append(_ frame: FrameData) {
        if frames.count < capacity {
            frames.append(frame)
        } else {
            frames[frameCount % capacity] = frame
        }
        frameCount += 1
    }

    /// Whether we have enough frames for inference.
    var isReady: Bool {
        return frames.count >= capacity
    }

    /// Whether we should run inference (every `stride` new frames once the buffer is full).
    func shouldRunInference(stride: Int = GVHMRConstants.inferenceStride) -> Bool {
        return isReady && (frameCount % stride == 0)
    }

    /// Get the last `capacity` frames in chronological order.
    func getWindow() -> [FrameData] {
        guard isReady else { return frames }
        var window = [FrameData]()
        window.reserveCapacity(capacity)
        let start = frameCount % capacity
        for i in 0..<capacity {
            window.append(frames[(start + i) % capacity])
        }
        return window
    }

    /// Pack keypoints into a flat array: [1, W, 17, 3].
    func packKeypoints() -> [Float] {
        let window = getWindow()
        var result = [Float]()
        result.reserveCapacity(capacity * 17 * 3)
        for frame in window {
            for kp in frame.keypoints {
                result.append(kp.x)
                result.append(kp.y)
                result.append(kp.z)  // confidence
            }
        }
        return result
    }

    /// Pack CLIFF camera params: [1, W, 3].
    func packCliffCam() -> [Float] {
        let window = getWindow()
        var result = [Float]()
        result.reserveCapacity(capacity * 3)
        for frame in window {
            result.append(frame.cliffCam.x)
            result.append(frame.cliffCam.y)
            result.append(frame.cliffCam.z)
        }
        return result
    }

    /// Pack camera angular velocity: [1, W, 6].
    func packCamAngvel() -> [Float] {
        let window = getWindow()
        var result = [Float]()
        result.reserveCapacity(capacity * 6)
        for frame in window {
            result.append(contentsOf: frame.camAngvel)
        }
        return result
    }

    /// Pack image features: [1, W, 1024].
    func packImageFeatures() -> [Float] {
        let window = getWindow()
        var result = [Float]()
        result.reserveCapacity(capacity * GVHMRConstants.imgseqDim)
        for frame in window {
            result.append(contentsOf: frame.imageFeatures)
        }
        return result
    }

    func reset() {
        frames.removeAll()
        frameCount = 0
    }
}
