import os
import re

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda
import tensorrt as trt
from PIL import Image
from tqdm import tqdm

# ---- adjust ----
ENGINE_PATH = "resnet34_veri776_int8.engine"
VERI_ROOT = "data/veri"
QUERY_LIST = os.path.join(VERI_ROOT, "name_query.txt")
GALLERY_LIST = os.path.join(VERI_ROOT, "name_test.txt")
QUERY_DIR = os.path.join(VERI_ROOT, "image_query")
GALLERY_DIR = os.path.join(VERI_ROOT, "image_test")
INPUT_H = 256
INPUT_W = 256
EMBED_DIM = 512  # ResNet-34 pre-fc dim
BATCH = 16  # must be <= MAX_BS used at build time
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
OUT_NPZ = "trt_int8_embeddings.npz"
# ----------------

# VeRi-776 filename: <vehicleID>_c<camID>_<timestamp>_<seq>.jpg
FN_RE = re.compile(r"(\d+)_c(\d+)_")


def parse_ids(path):
    m = FN_RE.search(os.path.basename(path))
    return int(m.group(1)), int(m.group(2))


def preprocess(pil_img):
    img = pil_img.convert("RGB").resize((INPUT_W, INPUT_H), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr.transpose(2, 0, 1)  # HWC -> CHW


class TRTInfer:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.in_name = self.out_name = None
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                self.in_name = n
            else:
                self.out_name = n

        self.stream = cuda.Stream()
        self.d_in = cuda.mem_alloc(BATCH * 3 * INPUT_H * INPUT_W * 4)
        self.d_out = cuda.mem_alloc(BATCH * EMBED_DIM * 4)

    def infer(self, batch):
        bs = batch.shape[0]
        self.context.set_input_shape(self.in_name, batch.shape)
        out = np.empty((bs, EMBED_DIM), dtype=np.float32)
        cuda.memcpy_htod_async(self.d_in, np.ascontiguousarray(batch), self.stream)
        self.context.set_tensor_address(self.in_name, int(self.d_in))
        self.context.set_tensor_address(self.out_name, int(self.d_out))
        self.context.execute_async_v3(self.stream.handle)
        cuda.memcpy_dtoh_async(out, self.d_out, self.stream)
        self.stream.synchronize()
        return out


def extract(infer, img_dir, name_list_path):
    with open(name_list_path) as f:
        names = [l.strip() for l in f if l.strip()]
    paths = [os.path.join(img_dir, n) for n in names]
    pids = np.array([parse_ids(p)[0] for p in paths])
    cids = np.array([parse_ids(p)[1] for p in paths])
    feats = np.empty((len(paths), EMBED_DIM), dtype=np.float32)

    for start in tqdm(range(0, len(paths), BATCH)):
        end = min(start + BATCH, len(paths))
        imgs = np.stack([preprocess(Image.open(p)) for p in paths[start:end]])
        feats[start:end] = infer.infer(imgs)

    # L2-normalize defensively (your engine may already do this)
    feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
    return feats, pids, cids


def evaluate(qf, gf, q_pids, g_pids, q_cids, g_cids, max_rank=10):
    """Standard VeRi-776 mAP + CMC with junk filtering."""
    distmat = 1.0 - qf @ gf.T  # cosine distance
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, None]).astype(np.int32)

    cmcs, APs = [], []
    for qi in range(len(q_pids)):
        order = indices[qi]
        # junk = same vehicle AND same camera
        junk = (g_pids[order] == q_pids[qi]) & (g_cids[order] == q_cids[qi])
        keep = ~junk
        m = matches[qi][keep]
        if not m.any():
            continue

        cmc = m.cumsum()
        cmc[cmc > 1] = 1
        cmcs.append(cmc[:max_rank])

        n_rel = m.sum()
        prec_at_k = m.cumsum() / np.arange(1, len(m) + 1)
        APs.append((prec_at_k * m).sum() / n_rel)

    cmc_curve = np.stack(cmcs).mean(0)
    return float(np.mean(APs)), cmc_curve


def main():
    infer = TRTInfer(ENGINE_PATH)
    print("extracting query embeddings...")
    qf, q_pids, q_cids = extract(infer, QUERY_DIR, QUERY_LIST)
    print("extracting gallery embeddings...")
    gf, g_pids, g_cids = extract(infer, GALLERY_DIR, GALLERY_LIST)

    print("computing mAP/CMC...")
    mAP, cmc = evaluate(qf, gf, q_pids, g_pids, q_cids, g_cids)
    print(f"mAP:     {mAP * 100:.2f}%")
    print(f"Rank-1:  {cmc[0] * 100:.2f}%")
    print(f"Rank-5:  {cmc[4] * 100:.2f}%")
    print(f"Rank-10: {cmc[9] * 100:.2f}%")

    np.savez(
        OUT_NPZ,
        qf=qf,
        gf=gf,
        q_pids=q_pids,
        g_pids=g_pids,
        q_cids=q_cids,
        g_cids=g_cids,
        mAP=mAP,
        cmc=cmc,
    )
    print(f"saved {OUT_NPZ}")


if __name__ == "__main__":
    main()
