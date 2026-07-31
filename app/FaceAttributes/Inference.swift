//  Inference.swift

import CoreML
import QuartzCore
import Vision

// order must match dataloader.py !
let ageBins = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
let genders = ["Male", "Female"]
let expressions = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]

let precisions = ["fp32", "fp16", "int8"]

struct FaceProbabilities {
    var box: CGRect   // vision coords, origin bottom left
    var expression: [Float]
    var age: [Float]
    var gender: [Float]
}

struct FacePrediction: Identifiable {
    let id = UUID()
    let box: CGRect   // layer coords
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

    func predict(_ pixelBuffer: CVPixelBuffer,
                 orientation: CGImagePropertyOrientation) -> [FaceProbabilities] {
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                            orientation: orientation, options: [:])

        let faceRequest = VNDetectFaceRectanglesRequest()
        try? handler.perform([faceRequest])

        var faces: [FaceProbabilities] = []
        for face in faceRequest.results ?? [] {
            let request = VNCoreMLRequest(model: model)
            request.imageCropAndScaleOption = .scaleFill
            request.regionOfInterest = widen(face.boundingBox)
            try? handler.perform([request])

            guard let results = request.results as? [VNCoreMLFeatureValueObservation] else { continue }
            faces.append(FaceProbabilities(box: face.boundingBox,
                                           expression: softmax(results, "expression"),
                                           age: softmax(results, "age"),
                                           gender: softmax(results, "gender")))
        }
        return faces
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
