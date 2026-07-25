//  Inference.swift

import CoreML
import Vision

struct FacePrediction: Identifiable {
    let id = UUID()
    let box: CGRect   // normalised, origin bottom left
    let age: String
    let gender: String
    let expression: String
    let ageConfidence: Float
    let genderConfidence: Float
    let expressionConfidence: Float
}

final class Inference {
    private let cropMargin = 0.25   // same margin as the training crops

    func predict(_ pixelBuffer: CVPixelBuffer) -> [FacePrediction] {
        return []
    }
}
