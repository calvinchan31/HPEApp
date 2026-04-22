import AVFoundation
import UIKit

/// Manages the camera capture session and provides frames to the pipeline.
class CameraManager: NSObject, ObservableObject {
    @Published var isRunning = false
    @Published var previewLayer: AVCaptureVideoPreviewLayer?
    @Published var isFrontCamera = false

    let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "com.gvhmr.camera", qos: .userInitiated)

    /// Callback for each new frame.
    var onFrame: ((CVPixelBuffer, CMTime) -> Void)?

    /// Camera intrinsics (focal length, principal point) for the active camera.
    private(set) var focalLength: Float = 1000
    private(set) var principalPoint: CGPoint = CGPoint(x: 360, y: 640)
    private(set) var imageSize: CGSize = CGSize(width: 720, height: 1280)
    private var imageSizeInitialized = false

    override init() {
        super.init()
        configureCaptureSession()
    }

    // MARK: - Setup

    private func configureCaptureSession() {
        session.beginConfiguration()
        session.sessionPreset = .hd1280x720

        // Camera input
        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                    for: .video,
                                                    position: .back),
              let input = try? AVCaptureDeviceInput(device: camera)
        else {
            print("ERROR: Cannot access camera")
            session.commitConfiguration()
            return
        }

        if session.canAddInput(input) {
            session.addInput(input)
        }

        // Extract intrinsics
        if let format = camera.activeFormat.formatDescription as CMFormatDescription? {
            let dims = CMVideoFormatDescriptionGetDimensions(format)
            imageSize = CGSize(width: CGFloat(dims.width), height: CGFloat(dims.height))
            // Approximate focal length from field of view
            let fov = camera.activeFormat.videoFieldOfView
            let fovRad = fov * .pi / 180
            focalLength = Float(dims.width) / (2 * tan(fovRad / 2))
            principalPoint = CGPoint(x: CGFloat(dims.width) / 2,
                                     y: CGFloat(dims.height) / 2)
        }

        // Video output
        videoOutput.setSampleBufferDelegate(self, queue: queue)
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]

        if session.canAddOutput(videoOutput) {
            session.addOutput(videoOutput)
        }

        // Set orientation to portrait
        if let connection = videoOutput.connection(with: .video) {
            if #available(iOS 17.0, *) {
                if connection.isVideoRotationAngleSupported(90) {
                    connection.videoRotationAngle = 90
                }
            } else {
                if connection.isVideoOrientationSupported {
                    connection.videoOrientation = .portrait
                }
            }
        }

        session.commitConfiguration()

        // Create preview layer
        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.videoGravity = .resizeAspectFill
        DispatchQueue.main.async {
            self.previewLayer = layer
        }
    }

    // MARK: - Control

    func start() {
        guard !isRunning else { return }
        queue.async { [weak self] in
            self?.session.startRunning()
            DispatchQueue.main.async {
                self?.isRunning = true
            }
        }
    }

    func stop() {
        guard isRunning else { return }
        queue.async { [weak self] in
            self?.session.stopRunning()
            DispatchQueue.main.async {
                self?.isRunning = false
            }
        }
    }

    /// Switch between front and back camera.
    func switchCamera() {
        let newPosition: AVCaptureDevice.Position = isFrontCamera ? .back : .front

        guard let newCamera = AVCaptureDevice.default(.builtInWideAngleCamera,
                                                       for: .video,
                                                       position: newPosition),
              let newInput = try? AVCaptureDeviceInput(device: newCamera)
        else {
            print("[Camera] Cannot access \(newPosition == .front ? "front" : "back") camera")
            return
        }

        session.beginConfiguration()

        // Remove existing camera input
        for input in session.inputs {
            session.removeInput(input)
        }

        if session.canAddInput(newInput) {
            session.addInput(newInput)
        }

        // Update intrinsics for new camera
        if let format = newCamera.activeFormat.formatDescription as CMFormatDescription? {
            let dims = CMVideoFormatDescriptionGetDimensions(format)
            let fov = newCamera.activeFormat.videoFieldOfView
            let fovRad = fov * .pi / 180
            focalLength = Float(dims.width) / (2 * tan(fovRad / 2))
        }

        // Set orientation + mirror for front camera
        if let connection = videoOutput.connection(with: .video) {
            if #available(iOS 17.0, *) {
                if connection.isVideoRotationAngleSupported(90) {
                    connection.videoRotationAngle = 90
                }
            } else {
                if connection.isVideoOrientationSupported {
                    connection.videoOrientation = .portrait
                }
            }
            connection.isVideoMirrored = (newPosition == .front)
        }

        session.commitConfiguration()

        // Reset intrinsics so captureOutput recalculates from actual buffer
        imageSizeInitialized = false

        DispatchQueue.main.async { [weak self] in
            self?.isFrontCamera = (newPosition == .front)
        }

        print("[Camera] Switched to \(newPosition == .front ? "front" : "back") camera")
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        // Get actual pixel buffer dimensions (accounts for video orientation rotation)
        if !imageSizeInitialized {
            let w = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
            let h = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
            imageSize = CGSize(width: w, height: h)
            principalPoint = CGPoint(x: w / 2, y: h / 2)
            // Recompute focal length for the rotated dimensions
            let shorterDim = Float(min(w, h))
            if focalLength > 0 {
                // focalLength was computed from landscape width; rescale to portrait width
                focalLength = focalLength * shorterDim / Float(max(w, h))
            }
            imageSizeInitialized = true
            print("[Camera] Actual buffer size: \(w)x\(h), focal=\(focalLength)")
        }

        let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        onFrame?(pixelBuffer, timestamp)
    }
}
