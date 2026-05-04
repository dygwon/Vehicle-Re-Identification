import os, glob, random
import numpy as np
from PIL import Image
import torchvision.transforms as T

# ---- adjust ----
VERI_TRAIN = "data/veri/image_train"
OUT_PATH   = "calib_data.npy"
N          = 1000
INPUT_H    = 256
INPUT_W    = 256
MEAN       = [0.485, 0.456, 0.406]
STD        = [0.229, 0.224, 0.225]
# ----------------

preprocess = T.Compose([
    T.Resize((INPUT_H, INPUT_W)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])

paths = sorted(glob.glob(os.path.join(VERI_TRAIN, "*.jpg")))
assert len(paths) >= N, f"only found {len(paths)} images"
random.seed(0)
random.shuffle(paths)
paths = paths[:N]

out = np.empty((N, 3, INPUT_H, INPUT_W), dtype=np.float32)
for i, p in enumerate(paths):
    img = Image.open(p).convert("RGB")
    out[i] = preprocess(img).numpy()

np.save(OUT_PATH, out)
print(f"saved {out.shape} ({out.nbytes/1e6:.1f} MB) -> {OUT_PATH}")
