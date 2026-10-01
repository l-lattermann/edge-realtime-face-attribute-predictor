//  Inference.swift

import CoreImage
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
    let roll: Double   // radians the face itself is tilted in the frame
    let landmarkCenter: CGPoint   // normalised, origin bottom left
    let landmarkSizePx: CGSize   // px, the raw hull of the landmark points
    let center: CGPoint   // normalised, centre of the extended box
    let sizePx: CGSize   // px, the extended box
    let squareSidePx: Double   // px, the square that is actually cropped
    let hullSizePx: CGSize   // px, the extended box turned upright again
    let cropBox: CGRect   // upright hull, the tracker matches on it
}

struct FacePrediction: Identifiable {
    let id = UUID()
    let blur: CGImage?   // the face blurred, nil unless the switch is on
    let blurBox: CGRect   // upright, the patch is not rolled
    let landmarkBox: CGRect   // layer coords, before the roll
    let box: CGRect   // the extended box
    let squareBox: CGRect   // what goes into the model
    let roll: Double   // radians the boxes turn with the face
    let age: String
    let gender: String
    let expression: String
}

final class Inference {
    // the landmark hull fills 0.82 of a training crop, these margins hit the same share
    private let foreheadMargin = 0.18
    private let chinMargin = 0.05
    private let sideMargin = 0.11
    private var model: VNCoreMLModel
    private(set) var precision = "fp16"

    private let context = CIContext()
    private(set) var lastCrop: CGImage?   // the square handed to the model

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

    // landmarks are cheap, run on every frame
    func detect(_ handler: VNImageRequestHandler, frame: CGSize) -> [DetectedFace] {
        let request = VNDetectFaceLandmarksRequest()
        try? handler.perform([request])

        var faces: [DetectedFace] = []
        for observation in request.results ?? [] {
            guard let all = observation.landmarks?.allPoints else { continue }

            // the face carries its own roll, how the phone is held does not matter
            let roll = observation.roll?.doubleValue ?? 0
            faces.append(hull(all.pointsInImage(imageSize: frame), roll, frame))
        }
        return faces
    }

    private func hull(_ points: [CGPoint], _ roll: Double, _ frame: CGSize) -> DetectedFace {
        var pivotX = 0.0
        var pivotY = 0.0
        for point in points {
            pivotX = pivotX + Double(point.x)
            pivotY = pivotY + Double(point.y)
        }
        pivotX = pivotX / Double(points.count)
        pivotY = pivotY / Double(points.count)

        var minU = Double.infinity, maxU = -Double.infinity
        var minV = Double.infinity, maxV = -Double.infinity
        for point in points {
            let dx = Double(point.x) - pivotX
            let dy = Double(point.y) - pivotY
            let u = dx * cos(-roll) - dy * sin(-roll)
            let v = dx * sin(-roll) + dy * cos(-roll)
            minU = min(minU, u); maxU = max(maxU, u)
            minV = min(minV, v); maxV = max(maxV, v)
        }

        let landmarkSizePx = CGSize(width: maxU - minU, height: maxV - minV)
        let midU = (minU + maxU) / 2
        let midV = (minV + maxV) / 2

        // grow the landmark box, v points at the forehead so that side gets the larger share
        let sizePx = CGSize(width: landmarkSizePx.width * (1 + 2 * sideMargin),
                            height: landmarkSizePx.height * (1 + foreheadMargin + chinMargin))
        let shiftV = landmarkSizePx.height * (foreheadMargin - chinMargin) / 2   // centre moves up

        // the model wants a square, so the longer side decides
        let squareSidePx = max(sizePx.width, sizePx.height)

        let landmarkCenter = place(midU, midV, pivotX, pivotY, roll, frame)
        let center = place(midU, midV + shiftV, pivotX, pivotY, roll, frame)

        // upright hull, only the tracker needs it
        let hullWidth = abs(sizePx.width * cos(roll)) + abs(sizePx.height * sin(roll))
        let hullHeight = abs(sizePx.width * sin(roll)) + abs(sizePx.height * cos(roll))
        let cropBox = CGRect(x: center.x - hullWidth / frame.width / 2,
                             y: center.y - hullHeight / frame.height / 2,
                             width: hullWidth / frame.width,
                             height: hullHeight / frame.height)

        return DetectedFace(roll: roll, landmarkCenter: landmarkCenter, landmarkSizePx: landmarkSizePx,
                            center: center, sizePx: sizePx, squareSidePx: squareSidePx,
                            hullSizePx: CGSize(width: hullWidth, height: hullHeight),
                            cropBox: cropBox)
    }

    // a point on the head axes back into normalised image coords
    private func place(_ u: Double, _ v: Double, _ pivotX: Double, _ pivotY: Double,
                       _ roll: Double, _ frame: CGSize) -> CGPoint {
        let x = pivotX + u * cos(roll) - v * sin(roll)
        let y = pivotY + u * sin(roll) + v * cos(roll)
        return CGPoint(x: x / frame.width, y: y / frame.height)
    }

    // the expensive half
    func classify(_ pixelBuffer: CVPixelBuffer, face: DetectedFace,
                  frame: CGSize) -> (String, String, String) {
        let request = VNCoreMLRequest(model: model)
        request.imageCropAndScaleOption = .scaleFill

        // a roi cannot be rotated, so the crop is cut here and the handler gets it alone
        let crop = upright(pixelBuffer, face, frame)
        lastCrop = context.createCGImage(crop, from: crop.extent)
        try? VNImageRequestHandler(ciImage: crop, options: [:]).perform([request])

        guard let results = request.results as? [VNCoreMLFeatureValueObservation] else {
            return ("?", "?", "?")
        }
        return (label(softmax(results, "age"), ageBins),
                label(softmax(results, "gender"), genders),
                label(softmax(results, "expression"), expressions))
    }

    // roll the face level and cut a square, the training crops are both
    private func upright(_ pixelBuffer: CVPixelBuffer, _ face: DetectedFace,
                         _ frame: CGSize) -> CIImage {
        let centerX = face.center.x * frame.width
        let centerY = face.center.y * frame.height
        let sidePx = face.squareSidePx

        // face centre to the origin, roll it level, then into the middle of the square
        var transform = CGAffineTransform(translationX: -centerX, y: -centerY)
        transform = transform.concatenating(CGAffineTransform(rotationAngle: -face.roll))
        transform = transform.concatenating(CGAffineTransform(translationX: sidePx / 2,
                                                              y: sidePx / 2))

        // clamped, a face at the edge would otherwise get a transparent border
        let image = CIImage(cvPixelBuffer: pixelBuffer).clampedToExtent().transformed(by: transform)
        return image.cropped(to: CGRect(x: 0, y: 0, width: sidePx, height: sidePx))
    }

    // a blur has no orientation, so the patch is cut upright and never rolled
    func blurPatch(_ pixelBuffer: CVPixelBuffer, face: DetectedFace, frame: CGSize) -> CGImage? {
        let width = face.hullSizePx.width
        let height = face.hullSizePx.height
        let x = face.center.x * frame.width - width / 2
        let y = face.center.y * frame.height - height / 2

        // sigma scales with the face so a near and a far face look equally blurred
        let image = CIImage(cvPixelBuffer: pixelBuffer).clampedToExtent()
        let blurred = image.applyingGaussianBlur(sigma: height / 50)
        return context.createCGImage(blurred, from: CGRect(x: x, y: y, width: width, height: height))
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
