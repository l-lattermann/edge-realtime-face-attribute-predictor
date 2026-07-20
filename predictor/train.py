import argparse
import csv
import torch
from sklearn.metrics import f1_score

import definition
import dataloader

LR = 0.001
FREEZE_EPOCHS = 2  # epochs the trunk stays frozen
DEVICE = "cuda"
ARTIFACTS_DIR = "artifacts"

parser = argparse.ArgumentParser()
parser.add_argument("--run-name", default="multitask_row9")
parser.add_argument("--branch-row", type=int, default=9)
parser.add_argument("--task", default="all", choices=["all", "expr", "age", "gender"])
parser.add_argument("--epochs", type=int, default=20)
args = parser.parse_args()

COLUMNS = ["epoch", "loss_age", "loss_gender", "loss_expr",
           "f1_age", "f1_gender", "f1_expr", "lr", "w_age", "w_gender", "w_expr"]

criterion = torch.nn.CrossEntropyLoss(ignore_index=dataloader.NO_LABEL)


def masked_loss(logits, labels, log_vars):
    # one ce per head, in the order the model returns
    losses = []
    for i in range(3):
        losses.append(criterion(logits[i], labels[i]))

    # kendall 2018: exp(-s)*L + s/2, s learned per task
    total = 0
    for i in range(3):
        total = total + torch.exp(-log_vars[i]) * losses[i] + log_vars[i] / 2
    return total, losses


def train_one_epoch(model, loader, optimiser, log_vars):
    model.train()
    sums = [0.0, 0.0, 0.0]

    for images, age, gender, expr in loader:
        images = images.to(DEVICE)
        labels = [expr.to(DEVICE), age.to(DEVICE), gender.to(DEVICE)]
        logits = model(images)

        total, losses = masked_loss(logits, labels, log_vars)
        optimiser.zero_grad()
        total.backward()
        optimiser.step()

        for i in range(3):
            sums[i] = sums[i] + losses[i].item()

    # order is expr, age, gender like the model returns
    return [s / len(loader) for s in sums]


def validate(model, loader):
    model.eval()
    true = [[], [], []]
    pred = [[], [], []]

    with torch.no_grad():
        for images, age, gender, expr in loader:
            logits = model(images.to(DEVICE))
            for i, labels in enumerate([expr, age, gender]):
                true[i].extend(labels.tolist())
                pred[i].extend(logits[i].argmax(1).cpu().tolist())

    # drop unlabelled, f1 over placeholders says nothing
    scores = []
    for i in range(3):
        pairs = []
        for t, p in zip(true[i], pred[i]):
            if t != dataloader.NO_LABEL:
                pairs.append([t, p])
        if len(pairs) == 0:
            scores.append(0.0)
            continue
        scores.append(f1_score([q[0] for q in pairs], [q[1] for q in pairs], average="macro"))
    return scores


model = definition.Predictor(args.branch_row).to(DEVICE)
log_vars = torch.zeros(3, requires_grad=True, device=DEVICE)
optimiser = torch.optim.AdamW(list(model.parameters()) + [log_vars], lr=LR)
schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, args.epochs)

train_loader = dataloader.make_loader("train", args.task)
val_loader = dataloader.make_loader("val", args.task)

# freeze trunk first, the random heads have to settle
for p in model.trunk.parameters():
    p.requires_grad = False

log = open(ARTIFACTS_DIR + "/" + args.run_name + "_train.csv", "w", newline="")
writer = csv.writer(log)
writer.writerow(COLUMNS)
best_f1 = 0

for epoch in range(1, args.epochs + 1):
    # unfreeze, from here everything trains
    if epoch == FREEZE_EPOCHS + 1:
        for p in model.trunk.parameters():
            p.requires_grad = True

    loss_expr, loss_age, loss_gender = train_one_epoch(model, train_loader, optimiser, log_vars)
    f1_expr, f1_age, f1_gender = validate(model, val_loader)
    schedule.step()

    weights = torch.exp(-log_vars).tolist()
    writer.writerow([epoch, loss_age, loss_gender, loss_expr, f1_age, f1_gender, f1_expr,
                     schedule.get_last_lr()[0], weights[1], weights[2], weights[0]])
    log.flush()  # write now, not at the end
    print(epoch, "f1", round(f1_expr, 3), round(f1_age, 3), round(f1_gender, 3))

    # keep the best, not the last
    mean_f1 = (f1_expr + f1_age + f1_gender) / 3
    if mean_f1 > best_f1:
        best_f1 = mean_f1
        torch.save(model.state_dict(), ARTIFACTS_DIR + "/" + args.run_name + ".pt")

log.close()
