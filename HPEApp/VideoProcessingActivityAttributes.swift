import Foundation

#if canImport(ActivityKit)
import ActivityKit

@available(iOS 16.1, *)
struct VideoProcessingActivityAttributes: ActivityAttributes {
    public struct ContentState: Codable, Hashable {
        var phase: String
        var detail: String
        var progress: Double
        var processedFrames: Int
        var totalFrames: Int
        var etaSeconds: Int
        var isMultiPerson: Bool
        var personCount: Int
        var isCancellable: Bool
        var isCompleted: Bool
    }

    var title: String
}
#endif
