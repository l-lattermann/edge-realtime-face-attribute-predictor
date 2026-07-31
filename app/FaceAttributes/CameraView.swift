//  CameraView.swift

import AVFoundation
import QuartzCore
import SwiftUI

final class CameraSession: NSObject, ObservableObject {
    @Published var predictions: [FacePrediction] = []
    @Published var latencyMs: Double = 0
    @Published var fps: Double = 0
    @Published var memoryMb: Double = 0
    @Published var usingFrontCamera = false
    @Published var highResolution = false
    @Published var smoothing = true
    @Published var precision = "fp16"

    let session = AVCaptureSession()
    let metrics = Metrics()
    var previewLayer: AVCaptureVideoPreviewLayer?

    private let output = AVCaptureVideoDataOutput()
    private let inference = Inference()
    private let smoother = Smoother()
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
        smoother.reset()
        queue.async { self.attachCamera() }
    }

    func toggleResolution() {
        highResolution.toggle()
        queue.async { self.applyPreset() }
    }

    func use(precision name: String) {
        precision = name
        smoother.reset()
        queue.async { self.inference.use(precision: name) }
    }

    private func configure() {
        session.beginConfiguration()
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String:
                                    kCVPixelFormatType_32BGRA]

        // frames during a running prediction get dropped, not queued
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: queue)
        if session.canAddOutput(output) {
            session.addOutput(output)
        }
        session.commitConfiguration()

        applyPreset()
        attachCamera()
        session.startRunning()
    }

    // 720p is enough for the 224 crop, 1080 only helps far away
    private func applyPreset() {
        session.beginConfiguration()
        session.sessionPreset = highResolution ? .hd1920x1080 : .hd1280x720
        session.commitConfiguration()
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
        if busy {
            return
        }
        busy = true

        // clock starts here, latency is the whole frame not only the model
        let frameStart = CACurrentMediaTime()

        let orientation: CGImagePropertyOrientation = usingFrontCamera ? .leftMirrored : .right
        let modelStart = CACurrentMediaTime()
        var faces = inference.predict(pixelBuffer, orientation: orientation)
        let modelMs = (CACurrentMediaTime() - modelStart) * 1000 / Double(max(faces.count, 1))

        if smoothing {
            faces = smoother.smooth(faces)
        }

        // main thread, the view reads these
        DispatchQueue.main.async {
            self.predictions = self.convert(faces)

            let latencyMs = (CACurrentMediaTime() - frameStart) * 1000
            self.latencyMs = latencyMs
            self.memoryMb = Metrics.memoryMb()
            self.fps = self.metrics.record(latencyMs: latencyMs, modelMs: modelMs,
                                           faceCount: faces.count, precision: self.precision,
                                           resolution: self.highResolution ? "1080p" : "720p")
            self.busy = false
        }
    }

    // vision box -> layer coords
    func convert(_ faces: [FaceProbabilities]) -> [FacePrediction] {
        var converted: [FacePrediction] = []
        for face in faces {
            var box = face.box
            if let layer = previewLayer {
                let flipped = CGRect(x: box.minX, y: 1 - box.maxY,
                                     width: box.width, height: box.height)
                box = layer.layerRectConverted(fromMetadataOutputRect: flipped)
            }
            converted.append(FacePrediction(box: box,
                                            age: label(face.age, ageBins),
                                            gender: label(face.gender, genders),
                                            expression: label(face.expression, expressions)))
        }
        return converted
    }
}

struct PreviewLayer: UIViewRepresentable {
    let camera: CameraSession

    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        let layer = AVCaptureVideoPreviewLayer(session: camera.session)
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
    @State private var shareItem: ShareItem?

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
                            .frame(width: 44, height: 44)
                            .background(.black.opacity(0.6))
                            .foregroundStyle(.white)
                            .clipShape(Circle())
                    }
                }

                Spacer()

                HStack(spacing: 10) {
                    Button(camera.highResolution ? "1080p" : "720p", action: camera.toggleResolution)

                    Button(camera.smoothing ? "smooth on" : "smooth off") {
                        camera.smoothing.toggle()
                    }

                    Menu(camera.precision) {
                        ForEach(precisions, id: \.self) { name in
                            Button(name) { camera.use(precision: name) }
                        }
                    }

                    Button("share \(camera.metrics.rowCount)") {
                        shareItem = ShareItem(url: camera.metrics.writeTemporaryFile())
                    }
                }
                .font(.system(.caption, design: .monospaced))
                .padding(8)
                .background(.black.opacity(0.6))
                .foregroundStyle(.white)
                .clipShape(Capsule())
            }
            .padding()
        }
        .preferredColorScheme(.dark)
        .onAppear { camera.start() }
        .sheet(item: $shareItem) { item in
            ShareSheet(url: item.url)
        }
    }
}

// sheet(item:) wants Identifiable, a url alone is not
struct ShareItem: Identifiable {
    let id = UUID()
    let url: URL
}
