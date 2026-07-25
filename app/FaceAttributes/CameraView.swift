//  CameraView.swift

import AVFoundation
import SwiftUI

final class CameraSession: NSObject, ObservableObject {
    @Published var predictions: [FacePrediction] = []
    @Published var latencyMs: Double = 0
    @Published var fps: Double = 0

    let session = AVCaptureSession()
    private let inference = Inference()
    private let metrics = Metrics()
    private let queue = DispatchQueue(label: "camera")

    func start() {
        // 720p is enough, bigger frames only cost crop time
        session.sessionPreset = .hd1280x720
    }
}

extension CameraSession: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
    }
}

struct PreviewLayer: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(layer)
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
            PreviewLayer(session: camera.session)
                .ignoresSafeArea()

            OverlayView(predictions: camera.predictions)

            VStack {
                Text(String(format: "%.0f fps   %.1f ms   %d faces",
                            camera.fps, camera.latencyMs, camera.predictions.count))
                    .font(.system(.caption, design: .monospaced))
                    .padding(6)
                    .background(.black.opacity(0.6))
                    .foregroundStyle(.white)
                Spacer()
            }
        }
        .onAppear { camera.start() }
    }
}
