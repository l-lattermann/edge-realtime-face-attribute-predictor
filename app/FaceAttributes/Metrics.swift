//  Metrics.swift

import Foundation
import QuartzCore

final class Metrics {
    private let window = 30   // fps average over this many frames
    private var recentMs: [Double] = []
    private var handle: FileHandle?
    private let startSeconds = CACurrentMediaTime()

    init() {
        // lands in documents, the Files app can reach it
        let folder = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let url = folder.appendingPathComponent("metrics.csv")
        let header = "seconds,latency_ms,fps,face_count,memory_mb,thermal\n"
        try? header.write(to: url, atomically: true, encoding: .utf8)
        handle = try? FileHandle(forWritingTo: url)
        handle?.seekToEndOfFile()
    }

    func record(latencyMs: Double, faceCount: Int) -> Double {
        recentMs.append(latencyMs)
        if recentMs.count > window {
            recentMs.removeFirst()
        }

        let meanMs = recentMs.reduce(0, +) / Double(recentMs.count)
        let fps = 1000 / meanMs

        let row = String(format: "%.2f,%.2f,%.1f,%d,%.1f,%d\n",
                         CACurrentMediaTime() - startSeconds, latencyMs, fps, faceCount,
                         Metrics.memoryMb(), ProcessInfo.processInfo.thermalState.rawValue)
        handle?.write(row.data(using: .utf8)!)
        return fps
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
