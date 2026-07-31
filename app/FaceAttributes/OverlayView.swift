//  OverlayView.swift

import SwiftUI

struct OverlayView: View {
    let predictions: [FacePrediction]

    var body: some View {
        ZStack {
            ForEach(predictions) { prediction in
                let box = prediction.box

                Rectangle()
                    .stroke(.green, lineWidth: 2)
                    .frame(width: box.width, height: box.height)
                    .position(x: box.midX, y: box.midY)

                VStack(alignment: .leading, spacing: 1) {
                    Text(prediction.age)
                    Text(prediction.gender)
                    Text(prediction.expression)
                }
                .font(.system(.caption2, design: .monospaced))
                .foregroundStyle(.white)
                .padding(4)
                .background(.black.opacity(0.6))
                .position(x: box.midX, y: box.maxY + 30)
            }
        }
    }
}
