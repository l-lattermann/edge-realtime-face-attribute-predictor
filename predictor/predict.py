import argparse
import csv
import os
import time
import torch

import definition
import dataloader

ARTIFACTS_DIR = "artifacts"

parser = argparse.ArgumentParser()
parser.add_argument("--run-name", default="multitask_row9")
parser.add_argument("--branch-row", type=int, default=9)
parser.add_argument("--backend", default="torch", choices=["torch", "coreml"])
parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "int8"])
parser.add_argument("--images", default="val")  # val, fairface_val, rafdb_test, or a folder
parser.add_argument("--batch-size", type=int, default=64)  # 1 for an honest latency number
parser.add_argument("--device", default="cuda")
parser.add_argument("--workers", type=int, default=4)
args = parser.parse_args()

COLUMNS = ["image", "true_age", "true_gender", "true_expr",
           "pred_age", "pred_gender", "pred_expr",
           "conf_age", "conf_gender", "conf_expr", "latency_ms"]


def collect(images):
    if images == "fairface_val":
        return dataloader.read_fairface("val")
    if images == "rafdb_test":
        return dataloader.read_rafdb("val")
    if images == "val":
        return dataloader.read_fairface("val") + dataloader.read_rafdb("val")

    # a folder of own photos, no ground truth
    no = dataloader.NO_LABEL
    samples = []
    for name in sorted(os.listdir(images)):
        samples.append([images + "/" + name, no, no, no, no])
    return samples


def load_model(path):
    state = torch.load(path, map_location="cpu")
    model = definition.Predictor(args.branch_row)
    model.load_state_dict(state)
    return model.to(args.device).eval()



def best(probabilities):
    confidence, index = probabilities.max(1)
    return index.cpu().tolist(), confidence.cpu().tolist()


def predict_torch(model, samples):
    loader = torch.utils.data.DataLoader(
        dataloader.FaceDataset(samples, dataloader.PLAIN),
        batch_size=args.batch_size, num_workers=args.workers)

    rows = []
    seen = 0
    with torch.no_grad():
        for images, age_lo, age_hi, gender, expr in loader:
            images = images.to(args.device)
            start = time.perf_counter()
            logits_expr, logits_age, logits_gender = model(images)
            if args.device == "cuda":
                torch.cuda.synchronize()
            # per image, so the number stays comparable across batch sizes
            latency_ms = (time.perf_counter() - start) * 1000 / len(images)

            pred_age, conf_age = best(logits_age.softmax(1))
            pred_gender, conf_gender = best(logits_gender.softmax(1))
            pred_expr, conf_expr = best(logits_expr.softmax(1))

            # a coarse age range cannot be scored against one bin, so it counts as no label
            true_age = torch.where(age_lo == age_hi, age_lo, torch.full_like(age_lo, dataloader.NO_LABEL))
            for i in range(len(images)):
                rows.append([samples[seen + i][0],
                             true_age[i].item(), gender[i].item(), expr[i].item(),
                             pred_age[i], pred_gender[i], pred_expr[i],
                             round(conf_age[i], 4), round(conf_gender[i], 4), round(conf_expr[i], 4),
                             round(latency_ms, 3)])
            seen = seen + len(images)
            if seen % (args.batch_size * 20) == 0:
                print(seen, "von", len(samples), flush=True)
    return rows


def predict_coreml(package, samples):
    import coremltools
    model = coremltools.models.MLModel(package)
    rows = []
    for path, age_lo, age_hi, gender, expr in samples:
        from PIL import Image
        image = Image.open(path).convert("RGB").resize((dataloader.IMAGE_SIZE_PX,) * 2)
        start = time.perf_counter()
        out = model.predict({"image": image})
        latency_ms = (time.perf_counter() - start) * 1000

        picked = []
        for head in ["age", "gender", "expression"]:
            logits = torch.tensor(out[head]).flatten()
            probabilities = logits.softmax(0)
            picked.append((int(probabilities.argmax()), round(float(probabilities.max()), 4)))
        age = age_lo if age_lo == age_hi else dataloader.NO_LABEL
        rows.append([path, age, gender, expr,
                     picked[0][0], picked[1][0], picked[2][0],
                     picked[0][1], picked[1][1], picked[2][1], round(latency_ms, 3)])
    return rows


samples = collect(args.images)
print(len(samples), "bilder,", args.images)

if args.backend == "torch":
    model = load_model(ARTIFACTS_DIR + "/" + args.run_name + ".pt")
    rows = predict_torch(model, samples)
else:
    package = ARTIFACTS_DIR + "/" + args.run_name + "_" + args.precision + ".mlpackage"
    rows = predict_coreml(package, samples)

out_path = ARTIFACTS_DIR + "/" + args.run_name + "_pred_" + args.backend + "_" + args.precision + ".csv"
with open(out_path, "w", newline="") as log:
    writer = csv.writer(log)
    writer.writerow(COLUMNS)
    writer.writerows(rows)
print("geschrieben:", out_path)
