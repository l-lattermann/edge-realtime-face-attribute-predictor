import argparse
import csv
import torch
from sklearn.metrics import f1_score

import definition
import dataloader

LR = 0.001
FREEZE_EPOCHS = 2  # epochs the trunk stays frozen
PATIENCE = 5  # stop after this many epochs without a better val f1
DEVICE = "cuda"
ARTIFACTS_DIR = "artifacts"
AGE = 1  # the model returns expr, age, gender

parser = argparse.ArgumentParser()
parser.add_argument("--run-name", default="multitask_row9")
parser.add_argument("--branch-row", type=int, default=9)
parser.add_argument("--task", default="all", choices=["all", "expr", "age", "gender"])
parser.add_argument("--epochs", type=int, default=20)
args = parser.parse_args()

NUM_AGE_BINS = len(dataloader.AGE_BINS)
COLUMNS = ["epoch", "loss_age", "loss_gender", "loss_expr",
           "f1_age", "f1_gender", "f1_expr", "lr", "w_age", "w_gender", "w_expr"]

bins = torch.arange(NUM_AGE_BINS, device=DEVICE)

# samples without a label for a head contribute nothing to that head
criterion = torch.nn.CrossEntropyLoss(ignore_index=dataloader.NO_LABEL)


def age_loss(logits, age_lo, age_hi):
    """Cross entropy over the group of bins that a label allows."""
    # fairface pins one bin, a raf-db range only says which group the age falls into
    keep = age_lo != dataloader.NO_LABEL
    logits, age_lo, age_hi = logits[keep], age_lo[keep], age_hi[keep]
    inside = (bins >= age_lo.view(-1, 1)) & (bins <= age_hi.view(-1, 1))

    # the mass that lands inside the group is everything the label allows
    log_p = torch.log_softmax(logits, 1)
    per_sample = -torch.logsumexp(log_p.masked_fill(~inside, -30.0), 1)

    return per_sample.mean()



def masked_loss(logits, age_lo, age_hi, gender, expr, log_vars):
    # a batch can have no label for a head at all -> ce over nothing is nan
    labelled = [(expr != dataloader.NO_LABEL).any(),
                (age_lo != dataloader.NO_LABEL).any(),
                (gender != dataloader.NO_LABEL).any()]
    losses = []
    for i in range(3):
        if not labelled[i]:
            losses.append(torch.zeros((), device=DEVICE))
        elif i == AGE:
            losses.append(age_loss(logits[i], age_lo, age_hi))
        elif i == 0:
            losses.append(criterion(logits[i], expr))
        else:
            losses.append(criterion(logits[i], gender))

    # kendall 2018: exp(-s)*L + s/2, s learned per task
    total = torch.zeros((), device=DEVICE)
    for i in range(3):
        if labelled[i]:
            total = total + torch.exp(-log_vars[i]) * losses[i] + log_vars[i] / 2
    return total, losses


def train_one_epoch(model, loader, optimiser, log_vars):
    model.train()
    sums = [0.0, 0.0, 0.0]

    for images, age_lo, age_hi, gender, expr in loader:
        images = images.to(DEVICE)
        age_lo, age_hi = age_lo.to(DEVICE), age_hi.to(DEVICE)
        gender, expr = gender.to(DEVICE), expr.to(DEVICE)
        logits = model(images)

        total, losses = masked_loss(logits, age_lo, age_hi, gender, expr, log_vars)
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
        for images, age_lo, age_hi, gender, expr in loader:
            logits = model(images.to(DEVICE))

            # a coarse range cannot be scored against one bin, only exact labels count
            exact = torch.where(age_lo == age_hi, age_lo, torch.full_like(age_lo, dataloader.NO_LABEL))
            for i, labels in enumerate([expr, exact, gender]):
                true[i].extend(labels.tolist())
                guess = logits[i].argmax(1)
                pred[i].extend(guess.cpu().tolist())

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
stale_epochs = 0

for epoch in range(1, args.epochs + 1):
    # unfreeze, from here everything trains
    if epoch == FREEZE_EPOCHS + 1:
        for p in model.trunk.parameters():
            p.requires_grad = True

    # read lr before the step, otherwise the row is one epoch off
    lr = schedule.get_last_lr()[0]
    loss_expr, loss_age, loss_gender = train_one_epoch(model, train_loader, optimiser, log_vars)
    f1_expr, f1_age, f1_gender = validate(model, val_loader)
    schedule.step()

    weights = torch.exp(-log_vars).tolist()
    writer.writerow([epoch, loss_age, loss_gender, loss_expr, f1_age, f1_gender, f1_expr,
                     lr, weights[1], weights[2], weights[0]])
    log.flush()  # write now, not at the end
    print(epoch, "f1", round(f1_expr, 3), round(f1_age, 3), round(f1_gender, 3), flush=True)

    # keep the best, not the last
    mean_f1 = (f1_expr + f1_age + f1_gender) / 3
    if mean_f1 > best_f1:
        best_f1 = mean_f1
        stale_epochs = 0
        torch.save(model.state_dict(), ARTIFACTS_DIR + "/" + args.run_name + ".pt")
    else:
        stale_epochs = stale_epochs + 1

    # checkpoint is allready the best, more epochs only cost time
    if stale_epochs >= PATIENCE:
        print("stopped after", epoch, "epochs, no improvement for", PATIENCE)
        break

log.close()
