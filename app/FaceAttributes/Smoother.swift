//  Smoother.swift

import CoreGraphics

final class Smoother {
    private let window = 5   // frames a face is averaged over
    private let maxDistance = 0.15   // max travel to still be the same face

    private var boxes: [CGRect] = []
    private var history: [[FaceProbabilities]] = []

    func reset() {
        boxes = []
        history = []
    }

    func smooth(_ faces: [FaceProbabilities]) -> [FaceProbabilities] {
        var nextBoxes: [CGRect] = []
        var nextHistory: [[FaceProbabilities]] = []
        var smoothed: [FaceProbabilities] = []

        for face in faces {
            // nearest centre wins. breaks when people swap places
            var match = -1
            var shortest = maxDistance
            for i in 0..<boxes.count {
                let dx = boxes[i].midX - face.box.midX
                let dy = boxes[i].midY - face.box.midY
                let distance = (dx * dx + dy * dy).squareRoot()
                if distance < shortest {
                    shortest = distance
                    match = i
                }
            }

            var track = match >= 0 ? history[match] : []
            track.append(face)
            if track.count > window {
                track.removeFirst()
            }

            nextBoxes.append(face.box)
            nextHistory.append(track)
            smoothed.append(FaceProbabilities(box: face.box,
                                              expression: mean(track) { $0.expression },
                                              age: mean(track) { $0.age },
                                              gender: mean(track) { $0.gender }))
        }

        boxes = nextBoxes
        history = nextHistory
        return smoothed
    }

    private func mean(_ track: [FaceProbabilities],
                      _ head: (FaceProbabilities) -> [Float]) -> [Float] {
        var total = head(track[0])
        for frame in track.dropFirst() {
            let probabilities = head(frame)
            for i in 0..<total.count where i < probabilities.count {
                total[i] = total[i] + probabilities[i]
            }
        }
        return total.map { $0 / Float(track.count) }
    }
}
