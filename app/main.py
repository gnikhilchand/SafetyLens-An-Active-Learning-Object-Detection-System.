from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, Response
import cv2
import numpy as np
from app.detector import SafetyDetector
import io
from PIL import Image

app = FastAPI(title="SafetyLens API", description="MLOps Pipeline for Object Detection")

# Initialize detector logic
detector = SafetyDetector(capture_threshold=0.4)

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "SafetyLens AI"}

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """
    Endpoint that receives an image, runs inference, 
    and triggers active learning storage if confidence is low.
    """
    try:
        image_bytes = await file.read()
        
        # Run the pipeline
        annotated_img, detections = detector.predict(image_bytes)

        # Convert annotated image back to bytes for return
        img_pil = Image.fromarray(annotated_img)
        buf = io.BytesIO()
        img_pil.save(buf, format="JPEG")
        processed_bytes = buf.getvalue()

        # Return the image directly (browsers can display this)
        return Response(content=processed_bytes, media_type="image/jpeg")
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/detect/json")
async def detect_objects_json(file: UploadFile = File(...)):
    """
    Alternative endpoint that returns JSON data instead of an image.
    """
    image_bytes = await file.read()
    _, detections = detector.predict(image_bytes)
    return {"detections": detections}