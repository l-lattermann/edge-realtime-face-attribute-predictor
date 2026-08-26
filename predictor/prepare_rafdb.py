import os
from PIL import Image

import dataloader

CROP_MARGIN = 0.25  # same margin as fairface
OUT_DIR = dataloader.RAFDB_DIR + "/cropped"

os.makedirs(OUT_DIR, exist_ok=True)

# one image per line, both splits mixed
for line in open(dataloader.RAFDB_DIR + "/EmoLabel/list_patition_label.txt"):
    name = line.split()[0]
    out_path = OUT_DIR + "/" + name
    if os.path.exists(out_path):
        continue

    box_file = dataloader.RAFDB_DIR + "/boundingbox/" + name.replace(".jpg", "_boundingbox.txt")
    x0, y0, x1, y1 = [float(v) for v in open(box_file).read().split()]
    margin_px = CROP_MARGIN * max(x1 - x0, y1 - y0)  # keeps hair and chin

    image = Image.open(dataloader.RAFDB_DIR + "/original/" + name).convert("RGB")
    image = image.crop((x0 - margin_px, y0 - margin_px, x1 + margin_px, y1 + margin_px))
    image = image.resize((dataloader.IMAGE_SIZE_PX, dataloader.IMAGE_SIZE_PX))
    image.save(out_path, quality=95)

print("cropped", len(os.listdir(OUT_DIR)), "images into", OUT_DIR)
