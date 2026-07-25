//  OverlayView.swift

import SwiftUI

struct OverlayView: View {
    let predictions: [FacePrediction]

    var body: some View {
        GeometryReader { geometry in
            ForEach(predictions) { prediction in
                // vision is normalised, origin bottom left
                let box = CGRect(
                    x: prediction.box.minX * geometry.size.width,
                    y: (1 - prediction.box.maxY) * geometry.size.height,
                    width: prediction.box.width * geometry.size.width,
                    height: prediction.box.height * geometry.size.height)

                Rectangle()
                    .stroke(.green, lineWidth: 2)
                    .frame(width: box.width, height: box.height)
                    .position(x: box.midX, y: box.midY)

                VStack(alignment: .leading, spacing: 2) {
                    label(prediction.age, prediction.ageConfidence)
                    label(prediction.gender, prediction.genderConfidence)
                    label(prediction.expression, prediction.expressionConfidence)
                }
                .position(x: box.midX, y: box.maxY + 28)
            }
        }
    }

    private func label(_ text: String, _ confidence: Float) -> some View {
        HStack(spacing: 4) {
            Text(text)
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.white)
            Capsule()
                .fill(.green)
                .frame(width: CGFloat(confidence) * 40, height: 4)   // 40pt at full conf
        }
    }
}
