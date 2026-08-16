//  Inference.swift

import CoreML
import QuartzCore
import Vision

// order must match dataloader.py !
let ageBins = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
let genders = ["Male", "Female"]
let expressions = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]

let precisions = ["fp32", "fp16", "int8"]

// measured along the head axes, not the image axes
struct DetectedFace {
    let center: CGPoint   // normalised, origin bottom left
    let sizePx: CGSize   // px, along the head axes
    let roll: Double   // radians, head tilt
    let cropBox: CGRect   // upright hull, this goes into core ml
}

struct FacePrediction: Identifiable {
    let id = UUID()
    let box: CGRect   // layer coords, before the tilt
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

    // landmarks give tilt + the real head size
    func detect(_ handler: VNImageRequestHandler, frame: CGSize) -> [DetectedFace] {
        let request = VNDetectFaceLandmarksRequest()
        try? handler.perform([request])

        var faces: [DetectedFace] = []
        for observation in request.results ?? [] {
            guard let landmarks = observation.landmarks,
                  let all = landmarks.allPoints else { continue }
            let box = observation.boundingBox

            // work in px, otherwise 720x1280 distorts the angles
            var xs: [Double] = []
            var ys: [Double] = []
            for point in all.normalizedPoints {
                xs.append((box.minX + Double(point.x) * box.width) * frame.width)
                ys.append((box.minY + Double(point.y) * box.height) * frame.height)
            }

            // eye line gives a smooth angle, observation.roll jumps
            let roll = eyeAngle(landmarks, box, frame)
            faces.append(headBox(xs, ys, roll, frame))
        }
        return faces
    }

    // angle eye to eye
    private func eyeAngle(_ landmarks: VNFaceLandmarks2D, _ box: CGRect, _ frame: CGSize) -> Double {
        guard let left = landmarks.leftEye, let right = landmarks.rightEye else { return 0 }
        let a = centre(left, box, frame)
        let b = centre(right, box, frame)
        return atan2(b.y - a.y, b.x - a.x)
    }

    // marks the mean point of one landmark region
    private func centre(_ region: VNFaceLandmarkRegion2D, _ box: CGRect, _ frame: CGSize) -> CGPoint {
        var sumX = 0.0
        var sumY = 0.0
        for point in region.normalizedPoints {
            sumX = sumX + (box.minX + Double(point.x) * box.width) * frame.width
            sumY = sumY + (box.minY + Double(point.y) * box.height) * frame.height
        }
        let count = Double(region.pointCount)
        return CGPoint(x: sumX / count, y: sumY / count)
    }

    private func headBox(_ xs: [Double], _ ys: [Double],
                         _ roll: Double, _ frame: CGSize) -> DetectedFace {
        var pivotX = 0.0
        var pivotY = 0.0
        for i in 0..<xs.count {
            pivotX = pivotX + xs[i]
            pivotY = pivotY + ys[i]
        }
        pivotX = pivotX / Double(xs.count)
        pivotY = pivotY / Double(ys.count)

        var minU = Double.infinity, maxU = -Double.infinity
        var minV = Double.infinity
        for i in 0..<xs.count {
            let dx = xs[i] - pivotX
            let dy = ys[i] - pivotY
            let u = dx * cos(-roll) - dy * sin(-roll)
            let v = dx * sin(-roll) + dy * cos(-roll)
            minU = min(minU, u); maxU = max(maxU, u)
            minV = min(minV, v)
        }

        // one unit = half face width, so the box is 2 wide and 3 high
        let unit = (maxU - minU) / 2
        let sizePx = CGSize(width: 2 * unit, height: 3 * unit)

        // lowest landmark is the chin, the box stands on it
        let bottomV = minV - 0.1 * unit
        let midU = (minU + maxU) / 2
        let midV = bottomV + 1.5 * unit

        // centre rotated back into image coords
        let centerX = pivotX + midU * cos(roll) - midV * sin(roll)
        let centerY = pivotY + midU * sin(roll) + midV * cos(roll)
        let center = CGPoint(x: centerX / frame.width, y: centerY / frame.height)

        // upright hull because a roi cannot be rotated
        let hullWidth = abs(sizePx.width * cos(roll)) + abs(sizePx.height * sin(roll))
        let hullHeight = abs(sizePx.width * sin(roll)) + abs(sizePx.height * cos(roll))
        let cropBox = CGRect(x: center.x - hullWidth / frame.width / 2,
                             y: center.y - hullHeight / frame.height / 2,
                             width: hullWidth / frame.width,
                             height: hullHeight / frame.height)

        return DetectedFace(center: center, sizePx: sizePx, roll: roll, cropBox: cropBox)
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
