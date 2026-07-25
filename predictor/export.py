import argparse
import csv
import os
import coremltools
import torch

import definition
import dataloader

IMAGE_SIZE_PX = 224
ARTIFACTS_DIR = "artifacts"

# ImageType has one scale for all 3 channels, the stds differ <1% so the mean is fine
STD_RGB_MEAN = sum(dataloader.STD_RGB) / 3

parser = argparse.ArgumentParser()
parser.add_argument("--run-name", default="multitask_row9")
parser.add_argument("--branch-row", type=int, default=9)
parser.add_argument("--random", action="store_true")  # untrained weights, only to test the path
args = parser.parse_args()


def to_coreml(model, precision):
    # trace records one forward, so eval first
    example = torch.rand(1, 3, IMAGE_SIZE_PX, IMAGE_SIZE_PX)
    traced = torch.jit.trace(model, example)

    # core ml does the normalisation, the app only gives the image
    image = coremltools.ImageType(
        name="image",
        shape=(1, 3, IMAGE_SIZE_PX, IMAGE_SIZE_PX),
        scale=1 / (255 * STD_RGB_MEAN),
        bias=[-m / STD_RGB_MEAN for m in dataloader.MEAN_RGB],
    )

    outputs = [coremltools.TensorType(name=n) for n in ["expression", "age", "gender"]]
    float_type = coremltools.precision.FLOAT32 if precision == "fp32" else coremltools.precision.FLOAT16

    converted = coremltools.convert(
        traced,
        inputs=[image],
        outputs=outputs,
        compute_units=coremltools.ComputeUnit.ALL,
        compute_precision=float_type,
        minimum_deployment_target=coremltools.target.iOS17,
    )

    # int8 = the fp16 model, weights quantised after
    if precision == "int8":
        config = coremltools.optimize.coreml.OptimizationConfig(
            global_config=coremltools.optimize.coreml.OpLinearQuantizerConfig(
                mode="linear_symmetric", dtype="int8"))
        converted = coremltools.optimize.coreml.linear_quantize_weights(converted, config=config)

    return converted


def package_size_mb(path):
    total_bytes = 0
    for folder, _, files in os.walk(path):
        for name in files:
            total_bytes = total_bytes + os.path.getsize(os.path.join(folder, name))
    return total_bytes / (1024 * 1024)


model = definition.Predictor(args.branch_row)

if not args.random:
    model.load_state_dict(torch.load(ARTIFACTS_DIR + "/" + args.run_name + ".pt"))
model.eval()

log = open(ARTIFACTS_DIR + "/" + args.run_name + "_export.csv", "w", newline="")
writer = csv.writer(log)
writer.writerow(["precision", "size_mb"])

for precision in ["fp32", "fp16", "int8"]:
    path = ARTIFACTS_DIR + "/" + args.run_name + "_" + precision + ".mlpackage"
    to_coreml(model, precision).save(path)
    size_mb = package_size_mb(path)
    writer.writerow([precision, round(size_mb, 2)])
    print(precision, round(size_mb, 2), "MB")

log.close()
