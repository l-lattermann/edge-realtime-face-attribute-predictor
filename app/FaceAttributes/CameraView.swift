//  CameraView.swift

import AVFoundation
import Combine
import QuartzCore
import SwiftUI
import Vision

final class CameraSession: NSObject, ObservableObject {
    @Published var predictions: [FacePrediction] = []
    @Published var latencyMs: Double = 0
    @Published var fps: Double = 0
    @Published var memoryMb: Double = 0
    @Published var usingFrontCamera = false
    @Published var highResolution = false
    @Published var inferEvery = 5   // classify every nth frame, detect on all
    @Published var precision = "fp16"

    let session = AVCaptureSession()
    let metrics = Metrics()
    var previewLayer: AVCaptureVideoPreviewLayer?

    private let output = AVCaptureVideoDataOutput()
    private let inference = Inference()
    private let tracker = Tracker()
    private let queue = DispatchQueue(label: "camera")
    private var busy = false
    private var frameCount = 0
    private let portraitAngle = 90.0   // sensor turn for portrait
    private let displayWindowSeconds = 1.0   // readout holds this long
    private var displaySince = CACurrentMediaTime()
    private var displayLatencySum = 0.0
    private var displayFpsSum = 0.0
    private var displayFrames = 0

    func start() {
        AVCaptureDevice.requestAccess(for: .video) { granted in
            if granted {
                self.queue.async { self.configure() }
            }
        }
    }

    func flipCamera() {
        usingFrontCamera.toggle()
        tracker.reset()
        queue.async { self.attachCamera() }
    }

    func toggleResolution() {
        highResolution.toggle()
        queue.async { self.applyPreset() }
    }

    func nextPrecision() {
        let current = precisions.firstIndex(of: precision) ?? 0
        let name = precisions[(current + 1) % precisions.count]
        precision = name
        tracker.reset()
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

        // startRunning blocks, so not on the delegate queue
        DispatchQueue.global().async {
            self.session.startRunning()
        }
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

        // sensor sits sideways, rotate here so vision and the layer share one coord system
        if let connection = output.connection(with: .video) {
            if connection.isVideoRotationAngleSupported(portraitAngle) {
                connection.videoRotationAngle = portraitAngle
            }
            connection.automaticallyAdjustsVideoMirroring = false
            connection.isVideoMirrored = usingFrontCamera
        }
        DispatchQueue.main.async {
            if let connection = self.previewLayer?.connection,
               connection.isVideoRotationAngleSupported(self.portraitAngle) {
                connection.videoRotationAngle = self.portraitAngle
            }
        }
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

        let frameSize = CGSize(width: CVPixelBufferGetWidth(pixelBuffer),
                               height: CVPixelBufferGetHeight(pixelBuffer))

        // clock starts here, latency is the whole frame not only the model
        let frameStart = CACurrentMediaTime()

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                            orientation: .up, options: [:])

        // detect every frame so the box does not stutter
        let boxes = inference.detect(handler, frame: frameSize)

        frameCount = frameCount + 1
        let classifying = frameCount % inferEvery == 0

        var faces: [(DetectedFace, String, String, String)] = []
        let modelStart = CACurrentMediaTime()

        if classifying {
            for face in boxes {
                let (age, gender, expression) = inference.classify(handler, box: face.cropBox)
                faces.append((face, age, gender, expression))
            }
            tracker.remember(faces)
        } else {
            // reuse the labels of the last classified frame
            for face in boxes {
                let carried = tracker.labels(for: face.cropBox) ?? ("", "", "")
                faces.append((face, carried.0, carried.1, carried.2))
            }
        }

        let modelMs = classifying
            ? (CACurrentMediaTime() - modelStart) * 1000 / Double(max(boxes.count, 1)) : 0

        // main thread, the view reads these
        DispatchQueue.main.async {
            self.predictions = self.convert(faces, frameSize)

            let latencyMs = (CACurrentMediaTime() - frameStart) * 1000
            let fps = self.metrics.record(latencyMs: latencyMs, modelMs: modelMs,
                                          classified: classifying, faceCount: boxes.count,
                                          precision: self.precision,
                                          resolution: self.highResolution ? "1080p" : "720p")
            self.publishReadout(latencyMs: latencyMs, fps: fps)
            self.busy = false
        }
    }

    private func publishReadout(latencyMs: Double, fps: Double) {
        displayLatencySum = displayLatencySum + latencyMs
        displayFpsSum = displayFpsSum + fps
        displayFrames = displayFrames + 1

        let elapsed = CACurrentMediaTime() - displaySince
        if elapsed < displayWindowSeconds {
            return
        }

        self.latencyMs = displayLatencySum / Double(displayFrames)
        self.fps = displayFpsSum / Double(displayFrames)
        self.memoryMb = Metrics.memoryMb()

        displayLatencySum = 0
        displayFpsSum = 0
        displayFrames = 0
        displaySince = CACurrentMediaTime()
    }

    // vision box -> layer coords
    func convert(_ faces: [(DetectedFace, String, String, String)], _ frame: CGSize) -> [FacePrediction] {
        let view = previewLayer?.bounds.size ?? .zero
        if view.width == 0 || frame.width == 0 {
            return []
        }

        // aspectFill covers both sides, so the bigger factor wins
        let scale = max(view.width / frame.width, view.height / frame.height)
        let shownWidth = frame.width * scale
        let shownHeight = frame.height * scale
        let offsetX = (view.width - shownWidth) / 2   // negative, the overhang is cropped
        let offsetY = (view.height - shownHeight) / 2

        var converted: [FacePrediction] = []
        for (face, age, gender, expression) in faces {
            // draws the upright hull, same rect that core ml gets
            let box = face.cropBox

            // vision counts from bottom left, the layer from top left
            let placed = CGRect(x: box.minX * shownWidth + offsetX,
                                y: (1 - box.maxY) * shownHeight + offsetY,
                                width: box.width * shownWidth,
                                height: box.height * shownHeight)
            converted.append(FacePrediction(box: placed, age: age,
                                            gender: gender, expression: expression))
        }
        return converted
    }
}

// backing layer IS the preview, so it allways has the right size
final class PreviewUIView: UIView {
    override class var layerClass: AnyClass {
        AVCaptureVideoPreviewLayer.self
    }

    var previewLayer: AVCaptureVideoPreviewLayer {
        layer as! AVCaptureVideoPreviewLayer
    }
}

struct PreviewLayer: UIViewRepresentable {
    let camera: CameraSession

    func makeUIView(context: Context) -> PreviewUIView {
        let view = PreviewUIView()
        view.previewLayer.session = camera.session
        view.previewLayer.videoGravity = .resizeAspectFill

        camera.previewLayer = view.previewLayer
        return view
    }

    func updateUIView(_ view: PreviewUIView, context: Context) {
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
                .ignoresSafeArea()

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

                    Button(camera.inferEvery == 1 ? "every frame" : "every \(camera.inferEvery)") {
                        camera.inferEvery = camera.inferEvery == 1 ? 5 : 1
                    }

                    Button(camera.precision) { camera.nextPrecision() }

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

struct ShareSheet: UIViewControllerRepresentable {
    let url: URL

    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: [url], applicationActivities: nil)
    }

    func updateUIViewController(_ controller: UIActivityViewController, context: Context) {
    }
}

// sheet(item:) wants Identifiable, a url alone is not
struct ShareItem: Identifiable {
    let id = UUID()
    let url: URL
}
