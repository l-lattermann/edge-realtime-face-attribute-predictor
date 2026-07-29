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

                VStack(alignment: .leading, spacing: 2) {
                    label(prediction.age, prediction.ageConfidence)
                    label(prediction.gender, prediction.genderConfidence)
                    label(prediction.expression, prediction.expressionConfidence)
                }
                .padding(4)
                .background(.black.opacity(0.6))
                .position(x: box.midX, y: box.maxY + 34)
            }
        }
    }

    private func label(_ text: String, _ confidence: Float) -> some View {
        HStack(spacing: 4) {
            Text(text)
                .font(.system(.caption2, design: .monospaced))
                .foregroundStyle(.white)
            Capsule()
                .fill(.green)
                .frame(width: CGFloat(confidence) * 40, height: 3)   // 40pt at full conf
        }
    }
}
