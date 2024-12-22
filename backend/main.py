from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import io
import os
import uvicorn
from PIL import Image
import torch
import numpy as np
from ultralytics import YOLO

app = FastAPI()

# Load the pre-trained model for object detection
model = YOLO("yolov8s.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device=device)

@app.get("/")
def server_message():
    return {"message": "Object detection server is running"}

# Endpoint for object detection
@app.post("/detect_objects/")
async def detect_objects(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        image_array = np.array(image)

        results = model.predict(image_array, conf=0.65, device=device)

        # Parsing the results
        detections = results[0]
        boxes = detections.boxes.xyxy.cpu().numpy().tolist()
        scores = detections.boxes.conf.cpu().numpy().tolist()
        classes = detections.boxes.cls.cpu().numpy().astype(int).tolist()
        class_names = [model.names[cls] for cls in classes]

        # Return the detection results as JSON response
        return JSONResponse({
            'boxes': boxes,
            'scores': scores,
            'classes': class_names
        })
    except Exception as e:
        print(f"Error during processing: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)
    
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)