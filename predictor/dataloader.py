import csv
import random
import torch
import torchvision
from PIL import Image

FAIRFACE_DIR = "../datasets/fairface"
RAFDB_DIR = "../datasets/raf_db"
IMAGE_SIZE_PX = 224
BATCH_SIZE = 64
NUM_WORKERS = 8
FAIRFACE_SHARE = 0.5  # rest is raf-db
CROP_MARGIN = 0.25  # fairface is allready margin025

# index order = logit order
AGE_BINS = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]
GENDERS = ["Male", "Female"]
EXPRESSIONS = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]

# -1 = no label, ignore_index skips it
NO_LABEL = -1

# imagenet stats, the backbone was trained with these
MEAN_RGB = [0.485, 0.456, 0.406]
STD_RGB = [0.229, 0.224, 0.225]

# gray + blur so the trunk cant tell the two sets apart by look
AUGMENT = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    torchvision.transforms.RandomGrayscale(0.2),
    torchvision.transforms.RandomApply([torchvision.transforms.GaussianBlur(5)], 0.2),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(MEAN_RGB, STD_RGB),
])

PLAIN = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(MEAN_RGB, STD_RGB),
])


def read_fairface(split):
    samples = []
    for row in csv.DictReader(open(FAIRFACE_DIR + "/fairface_label_" + split + ".csv")):
        path = FAIRFACE_DIR + "/" + row["file"]
        samples.append([path, None, AGE_BINS.index(row["age"]), GENDERS.index(row["gender"]), NO_LABEL])
    return samples


def read_rafdb(split):
    # both splits in one file, the prefix says which
    prefix = "train_" if split == "train" else "test_"
    samples = []
    for line in open(RAFDB_DIR + "/EmoLabel/list_patition_label.txt"):
        name, label = line.split()
        if not name.startswith(prefix):
            continue
        path = RAFDB_DIR + "/original/" + name
        box_file = RAFDB_DIR + "/boundingbox/" + name.replace(".jpg", "_boundingbox.txt")
        samples.append([path, box_file, NO_LABEL, NO_LABEL, int(label) - 1])  # file has 1..7
    return samples


def load_image(path, box_file, transform):
    image = Image.open(path).convert("RGB")

    # raf-db has full photos, widen the box like fairface
    if box_file is not None:
        x0, y0, x1, y1 = [float(v) for v in open(box_file).read().split()]
        margin_px = CROP_MARGIN * max(x1 - x0, y1 - y0)  # keeps hair and chin
        image = image.crop((x0 - margin_px, y0 - margin_px, x1 + margin_px, y1 + margin_px))

    image = image.resize((IMAGE_SIZE_PX, IMAGE_SIZE_PX))
    return transform(image)


class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, box_file, age, gender, expr = self.samples[i]
        return load_image(path, box_file, self.transform), age, gender, expr


def make_loader(split, task="all"):
    fairface = read_fairface(split) if task in ["all", "age", "gender"] else []
    rafdb = read_rafdb(split) if task in ["all", "expr"] else []
    samples = fairface + rafdb

    # no augment on val, has to stay comparable
    if split != "train":
        dataset = FaceDataset(samples, PLAIN)
        return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    # fixed ratio between both sets, no matter how big they are
    weights = []
    for i in range(len(samples)):
        share = FAIRFACE_SHARE if i < len(fairface) else 1 - FAIRFACE_SHARE
        pool = len(fairface) if i < len(fairface) else len(rafdb)
        weights.append(share / pool)

    sampler = torch.utils.data.WeightedRandomSampler(weights, len(samples), replacement=True)
    dataset = FaceDataset(samples, AUGMENT)
    return torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler,
                                       num_workers=NUM_WORKERS)


if __name__ == "__main__":
    for split in ["train", "val"]:
        fairface = read_fairface(split)
        rafdb = read_rafdb(split)
        print(split, "fairface", len(fairface), "rafdb", len(rafdb))

        # raf-db is very unbalanced
        for column, vocabulary, samples in [(2, AGE_BINS, fairface), (3, GENDERS, fairface),
                                            (4, EXPRESSIONS, rafdb)]:
            counts = [0] * len(vocabulary)
            for sample in samples:
                counts[sample[column]] = counts[sample[column]] + 1
            for name, count in zip(vocabulary, counts):
                print("   ", name, count)
