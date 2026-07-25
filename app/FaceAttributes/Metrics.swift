//  Metrics.swift

import Foundation

final class Metrics {
    private let window = 30   // fps average over this many frames
    private var recentMs: [Double] = []
    private var file: FileHandle?

    init() {
    }

    func record(latencyMs: Double, faceCount: Int) -> Double {
        recentMs.append(latencyMs)
        if recentMs.count > window {
            recentMs.removeFirst()
        }

        let meanMs = recentMs.reduce(0, +) / Double(recentMs.count)
        let fps = 1000 / meanMs

        return fps
    }
}
