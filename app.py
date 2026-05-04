import time

import cv2
import faiss
import numpy as np
import torch
from ultralytics import YOLO

# import tensorrt as trt # (Assuming you wrap your TRT engine, we'll use a PyTorch placeholder here for clarity)


class VectorDB:
    def __init__(self, vector_dim=512, ttl_seconds=86400):
        self.vector_dim = vector_dim
        self.ttl_seconds = ttl_seconds

        # FAISS index with ID mapping (IndexFlatL2 is simple Euclidean distance)
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(vector_dim))
        self.metadata = {}  # Maps faiss_id -> timestamp
        self.next_id = 0

    def add_vector(self, vector):
        vector = np.array([vector], dtype=np.float32)
        faiss.normalize_L2(vector)  # Normalize for cosine similarity equivalent
        self.index.add_with_ids(vector, np.array([self.next_id]))
        self.metadata[self.next_id] = time.time()
        self.next_id += 1

    def search(self, vector, threshold=0.6):
        if self.index.ntotal == 0:
            return False

        vector = np.array([vector], dtype=np.float32)
        faiss.normalize_L2(vector)
        distances, indices = self.index.search(vector, 1)  # Find Top 1

        # Lower distance is better in L2. Adjust threshold based on your model's embedding space.
        if distances[0][0] < threshold:
            return True  # Seen
        return False  # Not Seen

    def cleanup_expired(self):
        current_time = time.time()
        expired_ids = [
            vid
            for vid, ts in self.metadata.items()
            if current_time - ts > self.ttl_seconds
        ]

        if expired_ids:
            # FAISS allows removing IDs from IndexIDMap
            self.index.remove_ids(np.array(expired_ids))
            for vid in expired_ids:
                del self.metadata[vid]
            print(f"Cleaned up {len(expired_ids)} expired vehicles.")


class ResNetExtractor:
    def __init__(self, engine_path):
        # Load your INT8 TensorRT engine here.
        # For this script's readability, we simulate the inference step.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model = load_trt_engine(engine_path)

    def extract(self, crop_img):
        # 1. Preprocess: Resize to 224x224, convert to RGB, normalize
        img = cv2.resize(crop_img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1)) / 255.0
        img = np.expand_dims(img, axis=0).astype(np.float32)

        # 2. Inference (Replace with actual TRT execute logic)
        # tensor = torch.from_numpy(img).to(self.device)
        # embedding = self.model(tensor).cpu().numpy()[0]

        # Mock embedding for compilation
        embedding = np.random.rand(512).astype(np.float32)
        return embedding


def run_pipeline(video_source, yolo_engine_path, resnet_engine_path):
    # Initialize components
    detector = YOLO(yolo_engine_path, task="detect")
    extractor = ResNetExtractor(resnet_engine_path)
    db = VectorDB(vector_dim=512, ttl_seconds=86400)  # 24 hours

    cap = cv2.VideoCapture(video_source)

    # State tracking to avoid extracting features every single frame for the same car
    # Maps track_id -> timestamp of last feature extraction
    tracked_vehicles = {}
    EXTRACTION_COOLDOWN = 2.0  # Wait 2 seconds before sampling the same vehicle again
    CLEANUP_INTERVAL = 3600  # Run DB cleanup every hour
    last_cleanup = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()

        # 1. Run YOLOv8 tracking (uses ByteTrack by default)
        results = detector.track(
            frame, persist=True, classes=[2, 3, 5, 7], verbose=False
        )  # COCO classes: car, motorcycle, bus, truck

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box

                # Draw standard bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # 2. Check if we should extract features
                last_extracted = tracked_vehicles.get(track_id, 0)

                if current_time - last_extracted > EXTRACTION_COOLDOWN:
                    # Crop the vehicle
                    crop = frame[y1:y2, x1:x2]

                    if crop.size > 0:
                        # 3. Extract Features
                        embedding = extractor.extract(crop)

                        # 4. Search Index
                        seen = db.search(embedding, threshold=0.5)

                        if seen:
                            text = f"ID: {track_id} - SEEN"
                            color = (0, 0, 255)  # Red for already seen
                        else:
                            text = f"ID: {track_id} - NEW"
                            color = (0, 255, 0)  # Green for newly added
                            db.add_vector(embedding)

                        # Display Notification
                        cv2.putText(
                            frame,
                            text,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2,
                        )

                        # Update cooldown timer for this track_id
                        tracked_vehicles[track_id] = current_time

        # 5. Periodic Maintenance
        if current_time - last_cleanup > CLEANUP_INTERVAL:
            db.cleanup_expired()
            last_cleanup = current_time

        # Show feed
        cv2.imshow("Re-ID Pipeline", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    display_width=640,  # YOLO standard resolution is 640x640, so 640x480 is a good target
    display_height=480,
    framerate=30,
    flip_method=2,  # Change this (0, 1, 2, 3) if your camera is mounted upside down
):
    """
    Constructs a GStreamer pipeline string for the Jetson ISP.
    nvarguscamerasrc handles the hardware-accelerated CSI camera capture.
    """
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=true sync=false"
    )


if __name__ == "__main__":
    # Generate the GStreamer string
    video_src = gstreamer_pipeline(flip_method=0)

    print(f"Opening camera with GStreamer pipeline:\n{video_src}")

    # We must explicitly tell OpenCV to use the GStreamer backend (cv2.CAP_GSTREAMER)
    # Inside your run_pipeline function, ensure the VideoCapture call looks like this:
    # cap = cv2.VideoCapture(video_source, cv2.CAP_GSTREAMER)

    run_pipeline(video_src, "yolov8s_int8.engine", "resnet34_int8.engine")
