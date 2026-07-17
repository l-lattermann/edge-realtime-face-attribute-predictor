# mobilenetv3-small, howard 2019 table 2. branch after row 9, row 10 is the last stride-2
import torch
import torch.nn as nn
import torchvision


class Predictor(nn.Module):
    def __init__(self, branch_row=BRANCH_ROW):
        super().__init__()
        # loaded twice, every branch needs its own copy
        weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        rows_a = torchvision.models.mobilenet_v3_small(weights=weights).features
        rows_b = torchvision.models.mobilenet_v3_small(weights=weights).features

        # torchvision counts from 0, so row N = features[N-1]
        self.trunk = rows_a[:branch_row]
        self.branch_expr = rows_a[branch_row:]
        self.branch_face = rows_b[branch_row:]

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.head_expr = nn.Sequential(
            nn.Linear(POOLED_WIDTH, 256), nn.Hardswish(), nn.Dropout(0.2),
            nn.Linear(256, NUM_EXPRESSIONS),
        )
        self.head_age = nn.Sequential(
            nn.Linear(POOLED_WIDTH, 128), nn.Hardswish(), nn.Dropout(0.2),
            nn.Linear(128, NUM_AGE_BINS),
        )
        self.head_gender = nn.Sequential(
            nn.Linear(POOLED_WIDTH, 64), nn.Hardswish(), nn.Dropout(0.2),
            nn.Linear(64, NUM_GENDERS),
        )

    def forward(self, x):
        shared = self.trunk(x)

        expr = self.pool(self.branch_expr(shared)).flatten(1)
        face = self.pool(self.branch_face(shared)).flatten(1)

        # age + gender share one vector, expr has its own
        return self.head_expr(expr), self.head_age(face), self.head_gender(face)


def count_parameters(model):
    total = 0
    for p in model.parameters():
        total = total + p.numel()
    return total


if __name__ == "__main__":
    for branch_row in [5, 9]:
        model = Predictor(branch_row)
        print("row", branch_row, "params", count_parameters(model))
