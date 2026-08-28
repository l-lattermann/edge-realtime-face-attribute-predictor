import os
import zipfile
from PIL import Image

import dataloader

CROP_MARGIN = 0.25  # same margin as fairface
OUT_DIR = dataloader.RAFDB_DIR + "/cropped"
ANNOTATION_DIR = dataloader.RAFDB_DIR + "/Annotation/manual"


def square_crop(image, box):
    # square, so the resize to 224 does not squash the face along one axis
    x0, y0, x1, y1 = box
    side_px = (1 + 2 * CROP_MARGIN) * max(x1 - x0, y1 - y0)
    side_px = min(side_px, image.width, image.height)

    # slide the square back inside the image instead of padding the overhang with black
    left_px = min(max((x0 + x1) / 2 - side_px / 2, 0), image.width - side_px)
    top_px = min(max((y0 + y1) / 2 - side_px / 2, 0), image.height - side_px)
    return image.crop((left_px, top_px, left_px + side_px, top_px + side_px))


names = []
for line in open(dataloader.RAFDB_DIR + "/EmoLabel/list_patition_label.txt"):
    names.append(line.split()[0])

# crop every image once instead of once per epoch
os.makedirs(OUT_DIR, exist_ok=True)
for name in names:
    out_path = OUT_DIR + "/" + name
    if os.path.exists(out_path):
        continue

    box_file = dataloader.RAFDB_DIR + "/boundingbox/" + name.replace(".jpg", "_boundingbox.txt")
    box = [float(v) for v in open(box_file).read().split()]
    image = Image.open(dataloader.RAFDB_DIR + "/original/" + name).convert("RGB")
    image = square_crop(image, box)
    image = image.resize((dataloader.IMAGE_SIZE_PX, dataloader.IMAGE_SIZE_PX))
    image.save(out_path, quality=95)

print("cropped", len(os.listdir(OUT_DIR)), "images into", OUT_DIR)

# the manual annotation ships as one file per image, collect it into a single table
if not os.path.exists(ANNOTATION_DIR):
    zipfile.ZipFile(dataloader.RAFDB_DIR + "/Annotation/manual.zip").extractall(
        dataloader.RAFDB_DIR + "/Annotation")

table = open(dataloader.ATTRIBUTES_FILE, "w")
for name in names:
    # five landmark rows come first, then gender, race and age
    fields = open(ANNOTATION_DIR + "/" + name.replace(".jpg", "_manu_attri.txt")).read().split()
    table.write(name + " " + fields[-3] + " " + fields[-2] + " " + fields[-1] + "\n")
table.close()
print("wrote", dataloader.ATTRIBUTES_FILE)
