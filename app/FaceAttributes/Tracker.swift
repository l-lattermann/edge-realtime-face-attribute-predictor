//  Tracker.swift

import CoreGraphics

// remembers where each label was so it can follow the face
final class Tracker {
    private let maxDistance = 0.15   // max travel to still be the same face

    private var boxes: [CGRect] = []
    private var labels: [(String, String, String)] = []

    func reset() {
        boxes = []
        labels = []
    }

    func remember(_ predictions: [(DetectedFace, String, String, String)]) {
        boxes = predictions.map { $0.0.cropBox }
        labels = predictions.map { ($0.1, $0.2, $0.3) }
    }

    func labels(for box: CGRect) -> (String, String, String)? {
        // nearest centre wins. breaks when people swap places
        var match = -1
        var shortest = maxDistance
        for i in 0..<boxes.count {
            let dx = boxes[i].midX - box.midX
            let dy = boxes[i].midY - box.midY
            let distance = (dx * dx + dy * dy).squareRoot()
            if distance < shortest {
                shortest = distance
                match = i
            }
        }
        return match >= 0 ? labels[match] : nil
    }
}
