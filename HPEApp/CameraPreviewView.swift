import AVFoundation
import SwiftUI

/// UIViewRepresentable that wraps AVCaptureVideoPreviewLayer for camera feed display.
struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession
    var isFrontCamera: Bool = false

    func makeUIView(context: Context) -> UIView {
        let view = PreviewUIView()
        view.backgroundColor = .black
        let previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.frame = view.bounds
        view.layer.addSublayer(previewLayer)
        view.previewLayer = previewLayer
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {
        if let view = uiView as? PreviewUIView {
            view.previewLayer?.frame = view.bounds
            updatePreviewOrientation(for: view)
            // Mirror the preview layer for front camera so it feels natural
            if isFrontCamera {
                view.previewLayer?.connection?.automaticallyAdjustsVideoMirroring = false
                view.previewLayer?.connection?.isVideoMirrored = true
            } else {
                view.previewLayer?.connection?.automaticallyAdjustsVideoMirroring = true
            }
        }
    }

    private func updatePreviewOrientation(for view: UIView) {
        guard
            let connection = (view as? PreviewUIView)?.previewLayer?.connection,
            let orientation = view.window?.windowScene?.interfaceOrientation,
            let videoOrientation = AVCaptureVideoOrientation(interfaceOrientation: orientation)
        else {
            return
        }

        if #available(iOS 17.0, *) {
            if connection.isVideoRotationAngleSupported(videoOrientation.rotationAngle) {
                connection.videoRotationAngle = videoOrientation.rotationAngle
            }
        } else if connection.isVideoOrientationSupported {
            connection.videoOrientation = videoOrientation
        }
    }

    class PreviewUIView: UIView {
        var previewLayer: AVCaptureVideoPreviewLayer?

        override func layoutSubviews() {
            super.layoutSubviews()
            previewLayer?.frame = bounds
        }
    }
}

private extension AVCaptureVideoOrientation {
    init?(interfaceOrientation: UIInterfaceOrientation) {
        switch interfaceOrientation {
        case .portrait: self = .portrait
        case .portraitUpsideDown: self = .portraitUpsideDown
        case .landscapeLeft: self = .landscapeRight
        case .landscapeRight: self = .landscapeLeft
        default: return nil
        }
    }

    var rotationAngle: CGFloat {
        switch self {
        case .portrait: return 90
        case .portraitUpsideDown: return 270
        case .landscapeRight: return 180
        case .landscapeLeft: return 0
        @unknown default: return 90
        }
    }
}
