//  OverlayView.swift

import SwiftUI

// DEBUG REMOVE: draw every landmark as a dot
let showLandmarks = true

struct OverlayView: View {
    let predictions: [FacePrediction]
    let fullFrame: CGRect   // DEBUG REMOVE

    var body: some View {
        ZStack {
            // DEBUG REMOVE: must sit exactly on the visible image
            if showLandmarks {
                Rectangle()
                    .stroke(.white, lineWidth: 2)
                    .frame(width: fullFrame.width, height: fullFrame.height)
                    .position(x: fullFrame.midX, y: fullFrame.midY)
            }

            ForEach(predictions) { prediction in
                let box = prediction.box

                ZStack {
                    Rectangle()
                        .stroke(.green, lineWidth: 2)
                        .frame(width: box.width, height: box.height)

                    VStack(alignment: .leading, spacing: 1) {
                        Text(prediction.age)
                        Text(prediction.gender)
                        Text(prediction.expression)
                    }
                    .font(.system(.caption2, design: .monospaced))
                    .foregroundStyle(.white)
                    .padding(4)
                    .background(.black.opacity(0.6))
                    .offset(y: box.height / 2 + 26)   // under the box
                }
                .rotationEffect(.radians(-prediction.roll))   // vision turns the other way
                .position(x: box.midX, y: box.midY)

                // DEBUG REMOVE: vision box untouched, mapping error vs maths error
                if showLandmarks {
                    Rectangle()
                        .stroke(.blue, lineWidth: 2)
                        .frame(width: prediction.rawBox.width, height: prediction.rawBox.height)
                        .position(x: prediction.rawBox.midX, y: prediction.rawBox.midY)
                }

                // DEBUG REMOVE: landmarks allready in layer coords
                if showLandmarks {
                    ForEach(0..<prediction.landmarks.count, id: \.self) { i in
                        Circle()
                            .fill(i == 0 ? .red : .yellow)
                            .frame(width: 3, height: 3)
                            .position(prediction.landmarks[i])
                    }
                }
            }
        }
    }
}
