# calibrate-detection.py
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np, cv2, os
from dotenv import load_dotenv

load_dotenv()


EXPORT_DIR = os.getenv("EXPORT_DIR")

CALIB_DIR = 'calib_detect_images/'
ONNX_PATH = os.path.join(EXPORT_DIR, "runs", "detect", "coco-yolov8", "run-yolov8s-full", "weights", "best.onnx")
CACHE_PATH = 'calib_detect.cache'
INPUT_SHAPE = (3, 640, 640)   # CHW
BATCH_SIZE = 8


class Calibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, image_dir, cache_file, batch_size, input_shape):
        super().__init__()
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.shape = input_shape
        self.images = sorted([
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        self.idx = 0
        self.device_input = cuda.mem_alloc(
            batch_size * int(np.prod(input_shape)) * 4   # float32
        )
        print(f'calibrating with {len(self.images)} images, batch {batch_size}')

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.idx + self.batch_size > len(self.images):
            return None
        batch = []
        for p in self.images[self.idx:self.idx + self.batch_size]:
            img = cv2.imread(p)
            img = letterbox(img, self.shape[1])           # see note below
            img = img[:, :, ::-1].transpose(2, 0, 1)      # BGR->RGB, HWC->CHW
            img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
            batch.append(img)
        batch = np.ascontiguousarray(np.stack(batch))
        cuda.memcpy_htod(self.device_input, batch)
        self.idx += self.batch_size
        if self.idx % 80 == 0:
            print(f'  {self.idx}/{len(self.images)}')
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            print(f'using existing cache: {self.cache_file}')
            return open(self.cache_file, 'rb').read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, 'wb') as f:
            f.write(cache)
        print(f'wrote {self.cache_file} ({len(cache)} bytes)')


def letterbox(img, size):
    """resize+pad preserving aspect ratio - matches Ultralytics inference"""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (size - nh) // 2
    bottom = size - nh - top
    left = (size - nw) // 2
    right = size - nw - left
    return cv2.copyMakeBorder(img, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=(114, 114, 114))


# build network just enough to trigger calibration
logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

with open(ONNX_PATH, 'rb') as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError('ONNX parse failed')

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
config.set_flag(trt.BuilderFlag.INT8)
config.set_flag(trt.BuilderFlag.FP16)
config.int8_calibrator = Calibrator(CALIB_DIR, CACHE_PATH, BATCH_SIZE, INPUT_SHAPE)

# this triggers calibration as a side effect; we throw away the engine
print('running calibration...')
_ = builder.build_serialized_network(network, config)
print(f'done. cache at {CACHE_PATH}')
