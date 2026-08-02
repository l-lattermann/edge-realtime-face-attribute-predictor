//  Metrics.swift

import Foundation
import QuartzCore
import UIKit

final class Metrics {
    private let window = 30   // fps average over this many frames
    private var recentSeconds: [Double] = []
    private var rows: [String] = []
    private var lastFrameSeconds = CACurrentMediaTime()
    private let startSeconds = CACurrentMediaTime()

    let header = "seconds,latency_ms,model_ms,fps,classified,face_count,memory_mb,thermal,precision,resolution"

    func record(latencyMs: Double, modelMs: Double, classified: Bool, faceCount: Int,
                precision: String, resolution: String) -> Double {
        // fps from the gap between processed frames, not from the latency
        let now = CACurrentMediaTime()
        recentSeconds.append(now - lastFrameSeconds)
        lastFrameSeconds = now
        if recentSeconds.count > window {
            recentSeconds.removeFirst()
        }
        let meanSeconds = recentSeconds.reduce(0, +) / Double(recentSeconds.count)
        let fps = meanSeconds > 0 ? 1 / meanSeconds : 0

        rows.append(String(format: "%.2f,%.2f,%.2f,%.1f,%d,%d,%.1f,%d,%@,%@",
                           now - startSeconds, latencyMs, modelMs, fps, classified ? 1 : 0,
                           faceCount, Metrics.memoryMb(),
                           ProcessInfo.processInfo.thermalState.rawValue, precision, resolution))
        return fps
    }

    func writeTemporaryFile() -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("metrics.csv")
        let text = ([header] + rows).joined(separator: "\n")
        try? text.write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    var rowCount: Int {
        rows.count
    }

    // same number ios uses to decide what to kill
    static func memoryMb() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size
                                           / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        if result != KERN_SUCCESS {
            return 0
        }
        return Double(info.phys_footprint) / 1024 / 1024
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
