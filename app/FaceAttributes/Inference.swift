//  Inference.swift

import CoreML
import QuartzCore
import Vision

// order must match dataloader.py !
let ageBins = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
let genders = ["Male", "Female"]
let expressions = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]

let precisions = ["fp32", "fp16", "int8"]

// as vision sees it, before conversion
struct DetectedFace {
    let box: CGRect   // vision coords
    let roll: Double   // radians, head tilt
}

struct FacePrediction: Identifiable {
    let id = UUID()
    let box: CGRect   // layer coords
    let roll: Double
    let age: String
    let gender: String
    let expression: String
}

final class Inference {
    private let cropMargin = 0.25   // same margin as the training crops
    private var model: VNCoreMLModel
    private(set) var precision = "fp16"

    init() {
        model = Inference.load("fp16")
    }

    func use(precision name: String) {
        model = Inference.load(name)
        precision = name
    }

    private static func load(_ precision: String) -> VNCoreMLModel {
        let url = Bundle.main.url(forResource: "FaceAttributeModel_" + precision,
                                  withExtension: "mlmodelc")!
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all   // so it can go on the ANE
        return try! VNCoreMLModel(for: try! MLModel(contentsOf: url, configuration: configuration))
    }

    func detect(_ handler: VNImageRequestHandler) -> [DetectedFace] {
        let request = VNDetectFaceLandmarksRequest()
        try? handler.perform([request])

        var faces: [DetectedFace] = []
        for observation in request.results ?? [] {
            // vision box goes chin to eyebrows, so its centre is near the mouth
            var box = observation.boundingBox
            if let nose = observation.landmarks?.nose {
                var sumX = 0.0
                var sumY = 0.0
                for point in nose.normalizedPoints {
                    sumX = sumX + Double(point.x)
                    sumY = sumY + Double(point.y)
                }

                // landmarks are relative to the box, lift them into image coords
                let count = Double(nose.pointCount)
                let noseX = box.minX + sumX / count * box.width
                let noseY = box.minY + sumY / count * box.height
                box = box.offsetBy(dx: noseX - box.midX, dy: noseY - box.midY)
            }
            faces.append(DetectedFace(box: box, roll: observation.roll?.doubleValue ?? 0))
        }
        return faces
    }

    // the expensive half
    func classify(_ handler: VNImageRequestHandler, box: CGRect) -> (String, String, String) {
        // vision crops over the roi, no pixel copy here
        let request = VNCoreMLRequest(model: model)
        request.imageCropAndScaleOption = .scaleFill
        request.regionOfInterest = widen(box)
        try? handler.perform([request])

        guard let results = request.results as? [VNCoreMLFeatureValueObservation] else {
            return ("?", "?", "?")
        }
        return (label(softmax(results, "age"), ageBins),
                label(softmax(results, "gender"), genders),
                label(softmax(results, "expression"), expressions))
    }

    private func widen(_ box: CGRect) -> CGRect {
        let wide = box.insetBy(dx: -box.width * cropMargin, dy: -box.height * cropMargin)
        return wide.intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
    }

    private func softmax(_ results: [VNCoreMLFeatureValueObservation], _ name: String) -> [Float] {
        guard let array = results.first(where: { $0.featureName == name })?
            .featureValue.multiArrayValue else { return [] }

        // model gives logits, softmax here not in the graph
        var logits: [Float] = []
        for i in 0..<array.count {
            logits.append(array[i].floatValue)
        }
        let peak = logits.max() ?? 0
        let exponentials = logits.map { expf($0 - peak) }   // shifted, expf overflows otherwise
        let sum = exponentials.reduce(0, +)
        return exponentials.map { $0 / sum }
    }
}

func label(_ probabilities: [Float], _ vocabulary: [String]) -> String {
    if probabilities.isEmpty {
        return "?"
    }
    var best = 0
    for i in 0..<probabilities.count where probabilities[i] > probabilities[best] {
        best = i
    }
    return vocabulary[best]
}
