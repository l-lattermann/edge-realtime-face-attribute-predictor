//  CameraView.swift

import AVFoundation
import SwiftUI

final class CameraSession: NSObject, ObservableObject {
    @Published var predictions: [FacePrediction] = []
    @Published var latencyMs: Double = 0
    @Published var fps: Double = 0
    @Published var memoryMb: Double = 0
    @Published var usingFrontCamera = false

    let session = AVCaptureSession()
    var previewLayer: AVCaptureVideoPreviewLayer?
    private let output = AVCaptureVideoDataOutput()
    private let inference = Inference()
    private let metrics = Metrics()
    private let queue = DispatchQueue(label: "camera")
    private var busy = false

    func start() {
        AVCaptureDevice.requestAccess(for: .video) { granted in
            if granted {
                self.queue.async { self.configure() }
            }
        }
    }

    func flipCamera() {
        usingFrontCamera.toggle()
        queue.async { self.attachCamera() }
    }

    private func configure() {
        session.beginConfiguration()

        // 720p is enough for the 224 crop, 1080 only helps far away
        session.sessionPreset = .hd1280x720
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String:
                                    kCVPixelFormatType_32BGRA]

        // frames during a running prediction get dropped, not queued
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: queue)
        if session.canAddOutput(output) {
            session.addOutput(output)
        }
        session.commitConfiguration()

        attachCamera()
        session.startRunning()
    }

    private func attachCamera() {
        session.beginConfiguration()
        for input in session.inputs {
            session.removeInput(input)
        }

        let position: AVCaptureDevice.Position = usingFrontCamera ? .front : .back
        let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position)!
        let input = try! AVCaptureDeviceInput(device: device)
        if session.canAddInput(input) {
            session.addInput(input)
        }
        session.commitConfiguration()
    }
}

extension CameraSession: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        if busy { return }
        busy = true

        let orientation: CGImagePropertyOrientation = usingFrontCamera ? .leftMirrored : .right
        let (predictions, latencyMs) = inference.predict(pixelBuffer, orientation: orientation)
        let fps = metrics.record(latencyMs: latencyMs, faceCount: predictions.count)
        let memoryMb = Metrics.memoryMb()

        // main thread, the view reads these
        DispatchQueue.main.async {
            self.predictions = self.convert(predictions)
            self.latencyMs = latencyMs
            self.fps = fps
            self.memoryMb = memoryMb
            self.busy = false
        }
    }
}

extension CameraSession {
    // vision box -> layer coords
    func convert(_ predictions: [FacePrediction]) -> [FacePrediction] {
        guard let layer = previewLayer else { return predictions }

        var converted: [FacePrediction] = []
        for var prediction in predictions {
            let flipped = CGRect(x: prediction.box.minX, y: 1 - prediction.box.maxY,
                                 width: prediction.box.width, height: prediction.box.height)
            prediction.box = layer.layerRectConverted(fromMetadataOutputRect: flipped)
            converted.append(prediction)
        }
        return converted
    }
}

struct PreviewLayer: UIViewRepresentable {
    let camera: CameraSession

    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        let layer = AVCaptureVideoPreviewLayer(camera: camera)
        layer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(layer)

        camera.previewLayer = layer
        return view
    }

    func updateUIView(_ view: UIView, context: Context) {
        view.layer.sublayers?.first?.frame = view.bounds
    }
}

struct CameraView: View {
    @StateObject private var camera = CameraSession()

    var body: some View {
        ZStack {
            PreviewLayer(camera: camera)
                .ignoresSafeArea()

            OverlayView(predictions: camera.predictions)

            VStack {
                HStack(alignment: .top) {
                    Text(String(format: "%.0f fps\n%.1f ms\n%d faces\n%.0f MB",
                                camera.fps, camera.latencyMs,
                                camera.predictions.count, camera.memoryMb))
                        .font(.system(.caption, design: .monospaced))
                        .padding(6)
                        .background(.black.opacity(0.6))
                        .foregroundStyle(.white)

                    Spacer()

                    Button(action: camera.flipCamera) {
                        Image(systemName: "arrow.triangle.2.circlepath.camera")
                            .font(.title2)
                            .padding(10)
                            .background(.black.opacity(0.6))
                            .foregroundStyle(.white)
                            .clipShape(Circle())
                    }
                }
                Spacer()
            }
            .padding()
        }
        .preferredColorScheme(.dark)
        .onAppear { camera.start() }
    }
}
