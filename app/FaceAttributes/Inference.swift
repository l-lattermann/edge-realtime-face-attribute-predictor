//  Inference.swift

import CoreML
import QuartzCore
import Vision

// order must match dataloader.py !
let ageBins = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
let genders = ["Male", "Female"]
let expressions = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]

struct FacePrediction: Identifiable {
    let id = UUID()
    var box: CGRect   // vision coords until CameraSession converts them
    let age: String
    let gender: String
    let expression: String
    let ageConfidence: Float
    let genderConfidence: Float
    let expressionConfidence: Float
}

final class Inference {
    private let cropMargin = 0.25   // same margin as the training crops
    private let model: VNCoreMLModel

    init() {
        let url = Bundle.main.url(forResource: "FaceAttributeModel", withExtension: "mlmodelc")!
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all   // so it can go on the ANE
        model = try! VNCoreMLModel(for: try! MLModel(contentsOf: url, configuration: configuration))
    }

    func predict(_ pixelBuffer: CVPixelBuffer,
                 orientation: CGImagePropertyOrientation) -> ([FacePrediction], Double) {
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                            orientation: orientation, options: [:])

        let faceRequest = VNDetectFaceRectanglesRequest()
        try? handler.perform([faceRequest])
        let faces = faceRequest.results ?? []

        var predictions: [FacePrediction] = []
        let startSeconds = CACurrentMediaTime()

        for face in faces {
            let request = VNCoreMLRequest(model: model)
            request.imageCropAndScaleOption = .scaleFill
            request.regionOfInterest = widen(face.boundingBox)
            try? handler.perform([request])

            guard let results = request.results as? [VNCoreMLFeatureValueObservation] else { continue }
            let expr = best(results, "expression", expressions)
            let age = best(results, "age", ageBins)
            let gender = best(results, "gender", genders)

            predictions.append(FacePrediction(
                box: face.boundingBox,
                age: age.0, gender: gender.0, expression: expr.0,
                ageConfidence: age.1, genderConfidence: gender.1, expressionConfidence: expr.1))
        }

        // per face not per frame, stays comparable with more people
        let latencyMs = (CACurrentMediaTime() - startSeconds) * 1000 / Double(max(faces.count, 1))
        return (predictions, latencyMs)
    }

    private func widen(_ box: CGRect) -> CGRect {
        let marginX = box.width * cropMargin
        let marginY = box.height * cropMargin
        let wide = box.insetBy(dx: -marginX, dy: -marginY)
        return wide.intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
    }

    private func best(_ results: [VNCoreMLFeatureValueObservation],
                      _ name: String, _ vocabulary: [String]) -> (String, Float) {
        guard let array = results.first(where: { $0.featureName == name })?
            .featureValue.multiArrayValue else { return ("?", 0) }

        // model gives logits, softmax here not in the graph
        var logits: [Float] = []
        for i in 0..<array.count {
            logits.append(array[i].floatValue)
        }
        let peak = logits.max() ?? 0
        let exponentials = logits.map { expf($0 - peak) }   // shifted, expf overflows otherwise
        let sum = exponentials.reduce(0, +)

        var bestIndex = 0
        for i in 0..<exponentials.count where exponentials[i] > exponentials[bestIndex] {
            bestIndex = i
        }
        return (vocabulary[bestIndex], exponentials[bestIndex] / sum)
    }
}
