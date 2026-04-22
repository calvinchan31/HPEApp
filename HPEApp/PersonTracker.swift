import CoreGraphics

/// Multi-person tracker using greedy IoU matching across frames.
/// Assigns stable integer IDs to each tracked person.
///
/// Design mirrors the Python multi-HPE pipeline (YOLO tracking + IoU association).
/// Each frame, detections are matched to existing tracks by IoU.
/// Unmatched detections become new tracks; unmatched tracks are held for a few frames
/// then dropped.
class PersonTracker {

    struct TrackedPerson {
        let trackID: Int
        var bbox: CGRect
        var keypoints: [SIMD3<Float>]
        var confidence: Float
        var missedFrames: Int
    }

    /// Maximum frames a person can go undetected before being dropped.
    let maxMissedFrames: Int

    /// Maximum number of simultaneous persons.
    let maxPersons: Int

    /// Minimum IoU to match a detection to an existing track.
    let iouThreshold: Float = 0.2

    private(set) var tracks: [TrackedPerson] = []
    private var nextTrackID: Int = 0

    init(maxPersons: Int = 5, maxMissedFrames: Int = 15) {
        self.maxPersons = maxPersons
        self.maxMissedFrames = maxMissedFrames
    }

    func reset() {
        tracks.removeAll()
        nextTrackID = 0
    }

    /// Update tracks with new detections for the current frame.
    /// Returns the currently active tracked persons (sorted by track ID for stability).
    @discardableResult
    func update(detections: [(keypoints: [SIMD3<Float>], bbox: CGRect, confidence: Float)]) -> [TrackedPerson] {
        // Build IoU matrix: tracks × detections
        var matched = Set<Int>()      // indices into detections that got matched
        var matchedTracks = Set<Int>() // indices into tracks that got matched

        // Greedy matching: for each track, find best detection by IoU
        for (ti, track) in tracks.enumerated() {
            var bestIoU: Float = -1
            var bestDI: Int = -1
            for (di, det) in detections.enumerated() where !matched.contains(di) {
                let iou = computeIoU(track.bbox, det.bbox)
                if iou > bestIoU {
                    bestIoU = iou
                    bestDI = di
                }
            }

            if bestIoU >= iouThreshold && bestDI >= 0 {
                // Match found — update track
                tracks[ti].bbox = detections[bestDI].bbox
                tracks[ti].keypoints = detections[bestDI].keypoints
                tracks[ti].confidence = detections[bestDI].confidence
                tracks[ti].missedFrames = 0
                matched.insert(bestDI)
                matchedTracks.insert(ti)
            }
        }

        // Increment missed frames for unmatched tracks
        for ti in 0..<tracks.count where !matchedTracks.contains(ti) {
            tracks[ti].missedFrames += 1
        }

        // Remove tracks that exceeded maxMissedFrames
        tracks.removeAll { $0.missedFrames > maxMissedFrames }

        // Create new tracks from unmatched detections (up to maxPersons total)
        for (di, det) in detections.enumerated() where !matched.contains(di) {
            if tracks.count >= maxPersons { break }
            tracks.append(TrackedPerson(
                trackID: nextTrackID,
                bbox: det.bbox,
                keypoints: det.keypoints,
                confidence: det.confidence,
                missedFrames: 0
            ))
            nextTrackID += 1
        }

        // Return active tracks sorted by ID
        return tracks.filter { $0.missedFrames == 0 }.sorted { $0.trackID < $1.trackID }
    }

    /// Get the currently active (not missed) tracked persons.
    var activePersons: [TrackedPerson] {
        tracks.filter { $0.missedFrames == 0 }.sorted { $0.trackID < $1.trackID }
    }

    private func computeIoU(_ a: CGRect, _ b: CGRect) -> Float {
        let intersection = a.intersection(b)
        if intersection.isNull { return 0 }
        let interArea = intersection.width * intersection.height
        let unionArea = a.width * a.height + b.width * b.height - interArea
        guard unionArea > 0 else { return 0 }
        return Float(interArea / unionArea)
    }
}
