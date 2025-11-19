import cv2
import numpy as np
from ultralytics import YOLO
import os
import uuid
from datetime import datetime
from PIL import Image

class SafetyDetector:
    # def __init__(self, model_path='yolov8n.pt', capture_threshold=0.5):
    #     # Load the model (will download automatically if not present)
    #     self.model = YOLO(model_path)
    #     self.capture_threshold = capture_threshold
        
    #     # Simulating an S3 bucket locally
    #     self.data_lake_path = "data_lake/low_confidence_samples"
    #     os.makedirs(self.data_lake_path, exist_ok=True)
    def __init__(self, model_path='yolov8n.pt', capture_threshold=0.5):
        # Force a clean download/load by checking existence
        if not os.path.exists(model_path):
            print(f"Downloading {model_path}...")
        
        # Load the model
        self.model = YOLO(model_path)
        self.capture_threshold = capture_threshold
        
        # Simulating an S3 bucket locally
        self.data_lake_path = "data_lake/low_confidence_samples"
        os.makedirs(self.data_lake_path, exist_ok=True)

    def save_to_datalake(self, image: np.ndarray, conf: float):
        """
        Simulates an ETL process:
        If confidence is low, save raw image to S3/Disk for future labeling.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{self.data_lake_path}/review_{timestamp}_{unique_id}_conf_{conf:.2f}.jpg"
        
        # In a real job, you would use boto3 here to upload to AWS S3
        cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"[Active Learning] Low confidence ({conf:.2f}) detected. Saved to {filename}")

    def predict(self, image_bytes):
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = self.model(img_rgb)
        
        detections = []
        annotated_img = results[0].plot() # Draw boxes

        # Extract data and trigger Active Learning Pipeline
        for result in results:
            for box in result.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = self.model.names[cls]
                
                detections.append({
                    "label": label,
                    "confidence": round(conf, 2),
                    "box": box.xyxy[0].tolist()
                })

                # --- ENGINEERING TRIGGER ---
                # If the model is unsure, save it for the Data Team
                if conf < self.capture_threshold:
                    self.save_to_datalake(img_rgb, conf)

        return annotated_img, detections