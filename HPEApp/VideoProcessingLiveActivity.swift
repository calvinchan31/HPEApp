import Foundation

#if canImport(ActivityKit)
import ActivityKit
#endif

final class VideoProcessingLiveActivityManager {
    static let shared = VideoProcessingLiveActivityManager()

    private init() {}
    private var hasCompletedActivity = false

#if canImport(ActivityKit)
    private var activityStorage: Any?

    @available(iOS 16.2, *)
    private var activity: Activity<VideoProcessingActivityAttributes>? {
        get { activityStorage as? Activity<VideoProcessingActivityAttributes> }
        set { activityStorage = newValue }
    }

    @available(iOS 16.2, *)
    private var canRunLiveActivities: Bool {
        ActivityAuthorizationInfo().areActivitiesEnabled
    }
#endif

    func startOrUpdate(
        phase: String,
        detail: String = "",
        progress: Double,
        processedFrames: Int = 0,
        totalFrames: Int = 0,
        etaSeconds: Int = 0,
        isMultiPerson: Bool,
        personCount: Int,
        isCancellable: Bool
    ) {
#if canImport(ActivityKit)
    guard #available(iOS 16.2, *) else { return }
    guard canRunLiveActivities else { return }

        let safeProgress = min(max(progress, 0), 1)
        let state = VideoProcessingActivityAttributes.ContentState(
            phase: phase,
            detail: detail,
            progress: safeProgress,
            processedFrames: max(0, processedFrames),
            totalFrames: max(0, totalFrames),
            etaSeconds: max(0, etaSeconds),
            isMultiPerson: isMultiPerson,
            personCount: personCount,
            isCancellable: isCancellable,
            isCompleted: false
        )

        if let activity {
            Task {
                await activity.update(
                    ActivityContent(state: state, staleDate: Date().addingTimeInterval(120))
                )
            }
            return
        }

        let attributes = VideoProcessingActivityAttributes(title: "GVHMR Video")
        do {
            activity = try Activity.request(
                attributes: attributes,
                content: ActivityContent(state: state, staleDate: Date().addingTimeInterval(120)),
                pushType: nil
            )
            hasCompletedActivity = false
        } catch {
            // Keep app pipeline running even if Live Activities are unavailable.
            print("[LiveActivity] request failed: \(error)")
        }
#endif
    }

    func complete(finalPhase: String, progress: Double) {
#if canImport(ActivityKit)
        guard #available(iOS 16.2, *) else { return }
        guard let activity else { return }

        let finalState = VideoProcessingActivityAttributes.ContentState(
            phase: finalPhase,
            detail: "Ready to review results",
            progress: min(max(progress, 0), 1),
            processedFrames: 0,
            totalFrames: 0,
            etaSeconds: 0,
            isMultiPerson: false,
            personCount: 0,
            isCancellable: false,
            isCompleted: true
        )

        Task {
            await activity.update(
                ActivityContent(state: finalState, staleDate: Date().addingTimeInterval(3600))
            )
            self.hasCompletedActivity = true
        }
#endif
    }

    func clearCompletedIfNeeded() {
#if canImport(ActivityKit)
        guard #available(iOS 16.2, *) else { return }
        guard hasCompletedActivity else { return }
        guard let activity else { return }

        Task {
            await activity.end(nil, dismissalPolicy: .immediate)
            self.activity = nil
            self.hasCompletedActivity = false
        }
#endif
    }

    func end(finalPhase: String, progress: Double) {
#if canImport(ActivityKit)
        guard #available(iOS 16.2, *) else { return }
        guard let activity else { return }

        let finalState = VideoProcessingActivityAttributes.ContentState(
            phase: finalPhase,
            detail: "",
            progress: min(max(progress, 0), 1),
            processedFrames: 0,
            totalFrames: 0,
            etaSeconds: 0,
            isMultiPerson: false,
            personCount: 0,
            isCancellable: false,
            isCompleted: false
        )

        Task {
            await activity.end(
                ActivityContent(state: finalState, staleDate: nil),
                dismissalPolicy: .immediate
            )
            self.activity = nil
            self.hasCompletedActivity = false
        }
#endif
    }
}
