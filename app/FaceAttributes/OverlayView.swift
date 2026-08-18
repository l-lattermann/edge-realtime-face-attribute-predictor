//  OverlayView.swift

import SwiftUI

// DEBUG REMOVE: draw every landmark as a dot
let showLandmarks = true

struct OverlayView: View {
    let predictions: [FacePrediction]

    var body: some View {
        ZStack {
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
