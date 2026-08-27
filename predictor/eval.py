import argparse
import csv

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

import dataloader

ARTIFACTS_DIR = "artifacts"
ELDERLY_FROM_BIN = 7  # "60-69" and up, so the boundary sits at 60 years

parser = argparse.ArgumentParser()
parser.add_argument("--run-name", default="multitask_row9")
parser.add_argument("--backend", default="torch", choices=["torch", "coreml"])
parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "int8"])
args = parser.parse_args()

ATTRIBUTES = [("age", dataloader.AGE_BINS), ("gender", dataloader.GENDERS),
              ("expr", dataloader.EXPRESSIONS)]


def read_predictions(path, attribute):
    # a row only counts for a head that actually has a true label there
    pairs = []
    for row in csv.DictReader(open(path)):
        true = int(row["true_" + attribute])
        if true == dataloader.NO_LABEL:
            continue
        pairs.append([true, int(row["pred_" + attribute])])
    return pairs


def scores_per_class(pairs, vocabulary):
    true = [p[0] for p in pairs]
    pred = [p[1] for p in pairs]
    labels = list(range(len(vocabulary)))
    precision, recall, f1, support = precision_recall_fscore_support(
        true, pred, labels=labels, zero_division=0)
    rows = []
    for i, name in enumerate(vocabulary):
        rows.append([name, precision[i], recall[i], f1[i], int(support[i])])
    return rows


def collapse_age(pairs):
    # everything below the boundary is adult, the rest elderly
    return [[int(t >= ELDERLY_FROM_BIN), int(p >= ELDERLY_FROM_BIN)] for t, p in pairs]


def summarise(pairs):
    true = [p[0] for p in pairs]
    pred = [p[1] for p in pairs]
    return accuracy_score(true, pred), f1_score(true, pred, average="macro", zero_division=0)


path = ARTIFACTS_DIR + "/" + args.run_name + "_pred_" + args.backend + "_" + args.precision + ".csv"
out = open(ARTIFACTS_DIR + "/" + args.run_name + "_eval.csv", "w", newline="")
writer = csv.writer(out)
writer.writerow(["attribute", "class", "precision", "recall", "f1", "support"])

print(args.run_name, "  (" + args.backend + "/" + args.precision + ")")
for attribute, vocabulary in ATTRIBUTES:
    pairs = read_predictions(path, attribute)
    if not pairs:
        continue

    accuracy, macro_f1 = summarise(pairs)
    print()
    print("%-8s n=%d   acc %.4f   macro f1 %.4f" % (attribute, len(pairs), accuracy, macro_f1))

    for name, precision, recall, f1, support in scores_per_class(pairs, vocabulary):
        writer.writerow([attribute, name, round(precision, 4), round(recall, 4),
                         round(f1, 4), support])
        print("    %-14s p %.3f  r %.3f  f1 %.3f  n %5d" % (name, precision, recall, f1, support))
    writer.writerow([attribute, "ALL", "", "", round(macro_f1, 4), len(pairs)])

    if attribute == "age":
        # a neighbouring bin is a small mistake, plain accuracy does not show that
        near = sum(1 for t, p in pairs if abs(t - p) <= 1) / len(pairs)
        print("    %-14s %.4f" % ("+-1 bin acc", near))
        adult_elderly = collapse_age(pairs)
        accuracy, macro_f1 = summarise(adult_elderly)
        print("    %-14s acc %.4f  macro f1 %.4f" % ("adult/elderly", accuracy, macro_f1))
        for name, precision, recall, f1, support in scores_per_class(adult_elderly, ["adult", "elderly"]):
            writer.writerow(["age_collapsed", name, round(precision, 4), round(recall, 4),
                             round(f1, 4), support])

out.close()
print()
print("geschrieben:", ARTIFACTS_DIR + "/" + args.run_name + "_eval.csv")
