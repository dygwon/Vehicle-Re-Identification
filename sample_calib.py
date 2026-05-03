# sample_calib.py
import os
import random
import shutil
from collections import defaultdict

from dotenv import load_dotenv

load_dotenv()

EXPORT_DIR = os.getenv("EXPORT_DIR")

random.seed(42)

VAL_IMG = os.path.join(EXPORT_DIR, "images", "val")
VAL_LBL = os.path.join(EXPORT_DIR, "labels", "val")
DST = "calib_detect_images"
N_PER_CLASS = 75  # 4 classes * 75 = 300 target

os.makedirs(DST, exist_ok=True)

by_class = defaultdict(set)
all_imgs = [
    f for f in os.listdir(VAL_IMG) if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

for img_name in all_imgs:
    stem = os.path.splitext(img_name)[0]
    label_path = os.path.join(VAL_LBL, stem + ".txt")
    if not os.path.exists(label_path):
        continue
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                cls_id = int(parts[0])
                by_class[cls_id].add(img_name)

print("class -> available images:")
for cls_id in sorted(by_class):
    print(f"  class {cls_id}: {len(by_class[cls_id])}")

picked = set()
for cls_id, imgs in by_class.items():
    available = list(imgs - picked)
    take = min(N_PER_CLASS, len(available))
    picked.update(random.sample(available, take))

# top up to 300 if some classes were small
remaining = list(set(all_imgs) - picked)
random.shuffle(remaining)
while len(picked) < 300 and remaining:
    picked.add(remaining.pop())

for f in picked:
    shutil.copy(os.path.join(VAL_IMG, f), os.path.join(DST, f))

print(f"\ncopied {len(picked)} images to {DST}")
