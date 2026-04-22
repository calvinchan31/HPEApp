import SwiftUI
import WidgetKit

#if canImport(ActivityKit)
import ActivityKit

@available(iOSApplicationExtension 16.1, *)
struct GVHMRLiveActivityWidget: Widget {
    var body: some WidgetConfiguration {
        ActivityConfiguration(for: VideoProcessingActivityAttributes.self) { context in
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 8) {
                    Image(systemName: context.state.isCompleted ? "checkmark.seal.fill" : "figure.walk.motion")
                        .font(.caption.weight(.bold))
                        .foregroundStyle(context.state.isCompleted ? .green : .cyan)

                    Text(context.state.detail.isEmpty ? "GVHMR Video" : context.state.detail)
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                        .lineLimit(1)

                    Spacer()

                    Text("\(Int(context.state.progress * 100))%")
                        .font(.caption.monospacedDigit().weight(.bold))
                        .foregroundStyle(.primary)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(.white.opacity(0.14), in: Capsule())
                }

                Text(context.state.phase)
                    .font(.headline.weight(.semibold))
                    .lineLimit(1)

                ProgressView(value: context.state.progress)
                    .tint(context.state.isCompleted ? .green : .cyan)

                HStack(spacing: 8) {
                    if context.state.totalFrames > 0 {
                        Text("\(context.state.processedFrames)/\(context.state.totalFrames) frames")
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }

                    if context.state.isMultiPerson {
                        Label("\(max(context.state.personCount, 1)) tracked", systemImage: "person.3.fill")
                            .font(.caption2.weight(.semibold))
                            .foregroundStyle(.green)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(.green.opacity(0.15), in: Capsule())
                    }

                    Spacer()

                    if context.state.etaSeconds > 0 && !context.state.isCompleted {
                        Text("ETA \(formattedDuration(context.state.etaSeconds))")
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .padding(12)
            .background(
                LinearGradient(
                    colors: [Color.blue.opacity(0.18), Color.cyan.opacity(0.08)],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
            )
            .activityBackgroundTint(Color.black.opacity(0.92))
            .activitySystemActionForegroundColor(.cyan)
        } dynamicIsland: { context in
            DynamicIsland {
                DynamicIslandExpandedRegion(.leading) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("GVHMR")
                            .font(.caption.bold())
                        Text(context.state.isCompleted ? "Completed" : "Processing")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
                DynamicIslandExpandedRegion(.trailing) {
                    VStack(alignment: .trailing, spacing: 4) {
                        Text("\(Int(context.state.progress * 100))%")
                            .font(.caption.monospacedDigit().weight(.bold))
                        if context.state.etaSeconds > 0 && !context.state.isCompleted {
                            Text(formattedDuration(context.state.etaSeconds))
                                .font(.caption2.monospacedDigit())
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                DynamicIslandExpandedRegion(.bottom) {
                    VStack(alignment: .leading, spacing: 8) {
                        Text(context.state.detail.isEmpty ? "GVHMR Video" : context.state.detail)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)

                        Text(context.state.phase)
                            .font(.callout.weight(.semibold))
                            .lineLimit(1)

                        ProgressView(value: context.state.progress)
                            .tint(context.state.isCompleted ? .green : .cyan)

                        HStack {
                            if context.state.totalFrames > 0 {
                                Text("\(context.state.processedFrames)/\(context.state.totalFrames)")
                                    .font(.caption2.monospacedDigit())
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            if context.state.isMultiPerson {
                                Text("\(max(context.state.personCount, 1))P")
                                    .font(.caption2.weight(.semibold))
                                    .padding(.horizontal, 6)
                                    .padding(.vertical, 2)
                                    .background(.green.opacity(0.2), in: Capsule())
                            }
                        }
                    }
                }
            } compactLeading: {
                Image(systemName: context.state.isCompleted ? "checkmark.circle.fill" : "figure.walk.motion")
                    .foregroundStyle(context.state.isCompleted ? .green : .cyan)
            } compactTrailing: {
                if context.state.isCompleted {
                    Image(systemName: "checkmark")
                } else {
                    Text("\(Int(context.state.progress * 100))%")
                        .font(.caption2.monospacedDigit())
                }
            } minimal: {
                Image(systemName: context.state.isCompleted ? "checkmark.circle.fill" : "figure.walk.motion")
            }
            .widgetURL(URL(string: "gvhmr://video-processing"))
            .keylineTint(.cyan)
        }
    }

    private func formattedDuration(_ seconds: Int) -> String {
        let mins = seconds / 60
        let secs = seconds % 60
        if mins > 0 {
            return "\(mins)m \(secs)s"
        }
        return "\(secs)s"
    }
}

@available(iOSApplicationExtension 16.1, *)
@main
struct GVHMRLiveActivityBundle: WidgetBundle {
    var body: some Widget {
        GVHMRLiveActivityWidget()
    }
}
#endif

