import csv
import torch
import torchvision
from PIL import Image

FAIRFACE_DIR = "../datasets/fairface"
RAFDB_DIR = "../datasets/raf_db"
ATTRIBUTES_FILE = RAFDB_DIR + "/attributes.txt"
IMAGE_SIZE_PX = 224
BATCH_SIZE = 64
NUM_WORKERS = 12
FAIRFACE_SHARE = 0.5  # rest is raf-db

# index order = logit order
AGE_BINS = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "more than 70"]
GENDERS = ["Male", "Female"]
EXPRESSIONS = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]

# -1 = no label, ignore_index skips it
NO_LABEL = -1

# raf-db age is 5 coarse ranges, these are the fairface bins each one covers
RAFDB_AGE_SPAN = [[0, 1], [1, 2], [3, 4], [5, 7], [8, 8]]
# only 0-3 and 70+ are taken, the middle ranges guess at bins fairface allready fills
RAFDB_AGE_USED = [0, 4]
RAFDB_GENDER_UNSURE = 2

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


def read_fairface(split, task="all"):
    # age and gender sit in the same row, so a single task run has to blank the other one
    samples = []
    for row in csv.DictReader(open(FAIRFACE_DIR + "/fairface_label_" + split + ".csv")):
        age = AGE_BINS.index(row["age"]) if task in ["all", "age"] else NO_LABEL
        gender = GENDERS.index(row["gender"]) if task in ["all", "gender"] else NO_LABEL
        samples.append([FAIRFACE_DIR + "/" + row["file"], age, age, gender, NO_LABEL])
    return samples


def read_rafdb(split, task="all"):
    # gender and a coarse age range come from the manual annotation, prepare_rafdb.py collected it
    attributes = {}
    for line in open(ATTRIBUTES_FILE):
        name, gender, race, age_range = line.split()
        attributes[name] = [int(gender), int(age_range)]

    # both splits sit in one file, the prefix says which
    prefix = "train_" if split == "train" else "test_"
    samples = []
    for line in open(RAFDB_DIR + "/EmoLabel/list_patition_label.txt"):
        name, label = line.split()
        if not name.startswith(prefix):
            continue
        gender, age_range = attributes[name]

        # an unsure annotation is no annotation
        if gender == RAFDB_GENDER_UNSURE or task not in ["all", "gender"]:
            gender = NO_LABEL

        # the label names a group of bins, not one bin
        if age_range in RAFDB_AGE_USED and task in ["all", "age"]:
            age_lo, age_hi = RAFDB_AGE_SPAN[age_range]
        else:
            age_lo, age_hi = NO_LABEL, NO_LABEL

        expr = int(label) - 1 if task in ["all", "expr"] else NO_LABEL  # file has 1..7
        samples.append([RAFDB_DIR + "/cropped/" + name, age_lo, age_hi, gender, expr])
    return samples


def keep_labelled(samples):
    # a sample without a single label would only cost forward passes
    kept = []
    for sample in samples:
        if sample[1] != NO_LABEL or sample[3] != NO_LABEL or sample[4] != NO_LABEL:
            kept.append(sample)
    return kept


def load_image(path, transform):
    image = Image.open(path).convert("RGB")

    # both sets are allready at training size, usually a no-op
    if image.size != (IMAGE_SIZE_PX, IMAGE_SIZE_PX):
        image = image.resize((IMAGE_SIZE_PX, IMAGE_SIZE_PX))
    return transform(image)


class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, age_lo, age_hi, gender, expr = self.samples[i]
        return load_image(path, self.transform), age_lo, age_hi, gender, expr


def make_loader(split, task="all"):
    # both sources carry age and gender now, so every task reads both
    fairface = keep_labelled(read_fairface(split, task))
    rafdb = keep_labelled(read_rafdb(split, task))
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
        for column, vocabulary, samples in [(1, AGE_BINS, fairface), (3, GENDERS, fairface),
                                            (4, EXPRESSIONS, rafdb)]:
            counts = [0] * len(vocabulary)
            for sample in samples:
                counts[sample[column]] = counts[sample[column]] + 1
            for name, count in zip(vocabulary, counts):
                print("   ", name, count)

        # how much raf-db adds to the two thin age bins
        extra = [0] * len(AGE_BINS)
        for sample in rafdb:
            if sample[1] != NO_LABEL:
                extra[sample[1]] = extra[sample[1]] + 1
        print("    raf-db age ranges per first bin", extra)
