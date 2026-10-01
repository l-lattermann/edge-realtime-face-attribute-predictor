//  OverlayView.swift

import SwiftUI

struct OverlayView: View {
    let predictions: [FacePrediction]
    let blurFaces: Bool
    let debugBoxes: Bool

    var body: some View {
        ZStack {
            ForEach(predictions) { prediction in
                // debug shows every stage of the crop, otherwise only the box the model gets fed from
                if debugBoxes {
                    outline(prediction.landmarkBox, .green, prediction.roll)
                    outline(prediction.box, .orange, prediction.roll)
                    outline(prediction.squareBox, .red, prediction.roll)
                } else {
                    outline(prediction.box, .green, prediction.roll)
                }

                // the same pixels, gaussian blurred, laid back over the face
                if let patch = prediction.blur {
                    Image(decorative: patch, scale: 1, orientation: .up)
                        .resizable()
                        .frame(width: prediction.blurBox.width, height: prediction.blurBox.height)
                        .position(x: prediction.blurBox.midX, y: prediction.blurBox.midY)
                }

                ZStack {
                    VStack(alignment: .leading, spacing: 1) {
                        Text(prediction.age)
                        Text(prediction.gender)
                        Text(prediction.expression)
                    }
                    .font(.system(.caption2, design: .monospaced))
                    .foregroundStyle(.white)
                    .padding(4)
                    .background(.black.opacity(0.6))
                    .offset(y: prediction.box.height / 2 + 26)   // under the box
                }
                .rotationEffect(.radians(prediction.roll))
                .position(x: prediction.box.midX, y: prediction.box.midY)
            }
        }
    }

    // one box, rolled with the head and placed on its own centre
    private func outline(_ box: CGRect, _ color: Color, _ roll: Double) -> some View {
        Rectangle()
            .stroke(color, lineWidth: 2)
            .frame(width: box.width, height: box.height)
            .rotationEffect(.radians(roll))
            .position(x: box.midX, y: box.midY)
    }
}
