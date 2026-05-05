"""
Vehicle Re-Identification Pipeline for Jetson Orin Nano
=======================================================
- YOLOv8s INT8 TensorRT engine   (detection + ByteTrack via Ultralytics)
- ResNet-34 INT8 TensorRT engine (re-id features)
- FAISS IndexFlatIP gallery with 24h TTL  (cosine sim on L2-normalized vecs)

Tested target: JetPack 6.x  (TensorRT >= 8.5, CUDA 12), Python 3.10+

Dependencies:
    pip install ultralytics faiss-cpu opencv-python
    # tensorrt + pycuda are provided by JetPack
"""

import time
from collections import deque

import cv2
import faiss
import numpy as np
import pycuda.autoinit  # noqa: F401  -- creates/attaches the primary CUDA ctx
import pycuda.driver as cuda
import tensorrt as trt
from ultralytics import YOLO


# =============================================================================
# Configuration
# =============================================================================

# --- Engines ---------------------------------------------------------------
YOLO_ENGINE   = "engines/best_detect_fp16.engine"
RESNET_ENGINE = "engines/resnet34_veri776_int8.engine"

# --- Detection -------------------------------------------------------------
DETECT_CONF  = 0.35
MIN_BOX_AREA = 80 * 80           # ignore crops smaller than this

# --- Re-ID -----------------------------------------------------------------
REID_INPUT_HW = (256, 256)       # (H, W) -- must match how the engine was built
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)

# Cosine similarity threshold. 0.75-0.85 is typical; tune on your data.
MATCH_THRESHOLD = 0.78

# Don't re-attempt feature extraction for the same track until this many
# seconds have passed (only matters before a verdict is reached).
EXTRACTION_COOLDOWN = 0.5

# --- Gallery TTL -----------------------------------------------------------
TTL_SECONDS      = 24 * 3600
CLEANUP_INTERVAL = 600           # purge expired vectors every 10 min

# --- Tracker bookkeeping ---------------------------------------------------
TRACK_GC_INTERVAL = 30
TRACK_TTL         = 5.0          # forget a track this long after last sighting

# --- I/O -------------------------------------------------------------------
DISPLAY = True


# =============================================================================
# TensorRT runner
# =============================================================================

class TRTEngine:
    """Single-input / single-output TRT runner with pinned host buffers."""

    def __init__(self, engine_path: str):
        logger  = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load TRT engine {engine_path}")

        self.context = self.engine.create_execution_context()
        self.stream  = cuda.Stream()

        # TRT 8.5+ tensor-name API (replaces binding indices)
        self.in_name  = self.engine.get_tensor_name(0)
        self.out_name = self.engine.get_tensor_name(1)

        in_shape = tuple(self.engine.get_tensor_shape(self.in_name))
        if in_shape[0] == -1:                       # dynamic batch -> pin to 1
            in_shape = (1,) + in_shape[1:]
            self.context.set_input_shape(self.in_name, in_shape)
        out_shape = tuple(self.context.get_tensor_shape(self.out_name))

        self.in_shape, self.out_shape = in_shape, out_shape

        in_dtype  = trt.nptype(self.engine.get_tensor_dtype(self.in_name))
        out_dtype = trt.nptype(self.engine.get_tensor_dtype(self.out_name))

        self.h_in  = cuda.pagelocked_empty(int(np.prod(in_shape)),  in_dtype)
        self.h_out = cuda.pagelocked_empty(int(np.prod(out_shape)), out_dtype)
        self.d_in  = cuda.mem_alloc(self.h_in.nbytes)
        self.d_out = cuda.mem_alloc(self.h_out.nbytes)

        self.context.set_tensor_address(self.in_name,  int(self.d_in))
        self.context.set_tensor_address(self.out_name, int(self.d_out))

    def infer(self, x: np.ndarray) -> np.ndarray:
        np.copyto(self.h_in, x.ravel())
        cuda.memcpy_htod_async(self.d_in, self.h_in, self.stream)
        self.context.execute_async_v3(self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_out, self.d_out, self.stream)
        self.stream.synchronize()
        return self.h_out.reshape(self.out_shape).copy()


# =============================================================================
# Re-ID feature extractor
# =============================================================================

class ResNetExtractor:
    def __init__(self, engine_path: str):
        self.engine = TRTEngine(engine_path)
        # Embedding dimensionality = product of all non-batch dims of output.
        self.dim = int(np.prod(self.engine.out_shape[1:]))

    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        h, w = REID_INPUT_HW
        img = cv2.resize(crop_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)                       # HWC -> CHW
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return np.ascontiguousarray(img[None], dtype=np.float32)  # 1xCxHxW

    def extract(self, crop_bgr: np.ndarray) -> np.ndarray:
        feat = self.engine.infer(self._preprocess(crop_bgr)).reshape(-1)
        n = float(np.linalg.norm(feat)) + 1e-8
        return (feat / n).astype(np.float32)               # L2-normalized


# =============================================================================
# Vector DB with TTL
# =============================================================================

class VectorDB:
    """L2-normalized embeddings + IndexFlatIP -> inner product == cosine sim."""

    def __init__(self, dim: int, ttl: float = TTL_SECONDS):
        self.index    = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        self.metadata = {}      # id -> insertion timestamp
        self._next_id = 0
        self.ttl      = ttl

    def add(self, emb: np.ndarray) -> int:
        vid = self._next_id
        self._next_id += 1
        self.index.add_with_ids(
            emb.reshape(1, -1).astype(np.float32),
            np.array([vid], dtype=np.int64),
        )
        self.metadata[vid] = time.time()
        return vid

    def search(self, emb: np.ndarray, threshold: float = MATCH_THRESHOLD):
        if self.index.ntotal == 0:
            return False, 0.0, -1
        scores, ids = self.index.search(
            emb.reshape(1, -1).astype(np.float32), 1)
        score, vid = float(scores[0, 0]), int(ids[0, 0])
        return score >= threshold, score, vid

    def cleanup(self) -> int:
        now = time.time()
        expired = [v for v, ts in self.metadata.items() if now - ts > self.ttl]
        if not expired:
            return 0
        self.index.remove_ids(np.array(expired, dtype=np.int64))
        for v in expired:
            del self.metadata[v]
        return len(expired)


# =============================================================================
# Drawing
# =============================================================================

def draw_track(frame, box, state):
    x1, y1, x2, y2 = box
    verdict = state["verdict"]
    if verdict == "SEEN":
        color = (0, 0, 220)
        label = f"SEEN  cos={state['score']:.2f}"
    elif verdict == "NEW":
        color = (0, 200, 0)
        label = "NEW"
    else:
        color = (200, 180, 0)
        label = "..."

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
    cv2.putText(frame, label, (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


# =============================================================================
# Main pipeline
# =============================================================================

def run_pipeline(video_source, use_gstreamer=True):
    print("Loading YOLO engine...")
    detector = YOLO(YOLO_ENGINE, task="detect")

    print("Loading ResNet engine...")
    extractor = ResNetExtractor(RESNET_ENGINE)
    print(f"  -> embedding dim = {extractor.dim}")

    db = VectorDB(dim=extractor.dim)

    backend = cv2.CAP_GSTREAMER if use_gstreamer else cv2.CAP_ANY
    cap = cv2.VideoCapture(video_source, backend)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {video_source}")

    track_state   = {}     # tid -> {verdict, score, last_extracted, last_seen}
    last_cleanup  = time.time()
    last_track_gc = time.time()
    fps_buf       = deque(maxlen=30)

    print("Pipeline running. Press 'q' to quit.\n")

    while True:
        t0 = time.time()
        ok, frame = cap.read()
        if not ok:
            print("Stream ended.")
            break

        now = time.time()

        # --- 1. Detect + track --------------------------------------------------
        results = detector.track(
            frame,
            persist=True,
            conf=DETECT_CONF,
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]

        if results.boxes.id is not None:
            xyxy      = results.boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results.boxes.id.cpu().numpy().astype(int)

            for box, tid in zip(xyxy, track_ids):
                tid = int(tid)
                state = track_state.setdefault(tid, {
                    "verdict": None,
                    "score":   0.0,
                    "last_extracted": 0.0,
                    "last_seen":      now,
                })
                state["last_seen"] = now

                draw_track(frame, box, state)

                # --- 2. Decide whether to extract features for this track ------
                if state["verdict"] is not None:
                    continue                                     # already decided
                if now - state["last_extracted"] < EXTRACTION_COOLDOWN:
                    continue
                x1, y1, x2, y2 = box
                if (x2 - x1) * (y2 - y1) < MIN_BOX_AREA:
                    continue

                # Clamp to frame bounds (boxes can run a couple px over)
                h, w = frame.shape[:2]
                cx1, cy1 = max(0, x1), max(0, y1)
                cx2, cy2 = min(w, x2), min(h, y2)
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue

                # --- 3. Embed + search/insert ---------------------------------
                emb = extractor.extract(crop)
                seen, score, _ = db.search(emb)
                state["last_extracted"] = now
                state["score"] = score

                if seen:
                    state["verdict"] = "SEEN"
                else:
                    db.add(emb)
                    state["verdict"] = "NEW"

        # --- 4. Maintenance -----------------------------------------------------
        if now - last_cleanup > CLEANUP_INTERVAL:
            removed = db.cleanup()
            if removed:
                print(f"[gallery] removed {removed} expired vectors "
                      f"(remaining: {db.index.ntotal})")
            last_cleanup = now

        if now - last_track_gc > TRACK_GC_INTERVAL:
            stale = [tid for tid, s in track_state.items()
                     if now - s["last_seen"] > TRACK_TTL]
            for tid in stale:
                del track_state[tid]
            last_track_gc = now

        # --- 5. HUD -------------------------------------------------------------
        fps_buf.append(time.time() - t0)
        fps = len(fps_buf) / max(sum(fps_buf), 1e-6)
        cv2.putText(frame,
                    f"{fps:5.1f} FPS  |  gallery: {db.index.ntotal}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2)

        if DISPLAY:
            cv2.imshow("Re-ID Pipeline", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


# =============================================================================
# CSI camera GStreamer pipeline
# =============================================================================

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280, capture_height=720,
    display_width=960,  display_height=540,
    framerate=30, flip_method=0,
):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={capture_width}, "
        f"height={capture_height}, framerate={framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width={display_width}, height={display_height}, "
        f"format=BGRx ! videoconvert ! video/x-raw, format=BGR ! "
        f"appsink drop=true sync=false max-buffers=1"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vehicle Re-ID pipeline")
    parser.add_argument("--source", "-s", default=None,
                        help="Path to a video file. If omitted, uses the CSI "
                             "camera via GStreamer.")
    parser.add_argument("--flip", type=int, default=0,
                        help="GStreamer flip-method (0-3) for the CSI camera.")
    args = parser.parse_args()

    if args.source:
        print(f"Reading from file: {args.source}\n")
        run_pipeline(args.source, use_gstreamer=False)
    else:
        src = gstreamer_pipeline(flip_method=args.flip)
        print("GStreamer pipeline:\n" + src + "\n")
        run_pipeline(src, use_gstreamer=True)
