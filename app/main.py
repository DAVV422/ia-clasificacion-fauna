import logging
from pathlib import Path
from typing import List, Set
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.config import (
    API_TITLE,
    API_VERSION,
    MEGADETECTOR_URL,
    MEGADETECTOR_PATH,
    TEMP_DIR,
    DETECTIONS_DIR,
    VIDEO_FPS_SAMPLING
)
from app.utils import download_megadetector_weights, save_uploaded_file, clean_up_file
from app.detector import WildlifeDetector
from app.schemas import AnalysisResponse, TimestampDetection, DetectionDetail

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description="FastAPI service for animal detection using Microsoft MegaDetector v5a with local crop storage."
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector
detector = WildlifeDetector()

# Mount the detections folder so crops can be accessed as static files
# E.g. http://localhost:8000/detections/animal/uuid.jpg
app.mount("/detections", StaticFiles(directory=str(DETECTIONS_DIR)), name="detections")

@app.on_event("startup")
def startup_event():
    """
    Bootstrap operation on startup:
    1. Ensures MegaDetector v5a weights are downloaded.
    2. Warmup the PyTorch model so the first request is instant.
    """
    logger.info("Initializing application startup sequence...")
    # 1. Download weights if missing
    download_megadetector_weights(MEGADETECTOR_URL, MEGADETECTOR_PATH)
    
    # 2. Load model into memory
    try:
        detector.load_model()
    except Exception as e:
        logger.error(f"Failed to load MegaDetector model during startup: {e}")

@app.get("/")
def read_root():
    return {
        "service": API_TITLE,
        "version": API_VERSION,
        "status": "healthy",
        "model_loaded": detector.model is not None,
        "species_classifier_loaded": detector.classifier is not None,
        "device": detector.device
    }


def is_image_file(filename: str) -> bool:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    return Path(filename).suffix.lower() in extensions

def is_video_file(filename: str) -> bool:
    extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
    return Path(filename).suffix.lower() in extensions

@app.post("/api/v1/detect", response_model=AnalysisResponse)
async def detect_wildlife(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Analyzes uploaded image or video using MegaDetector v5a.
    - Crops any detected animals/persons/vehicles.
    - Saves cropped images in folders organized by category.
    - Returns timestamps, coordinates, and crop URLs.
    """
    filename = file.filename
    logger.info(f"Received detection request for file: {filename}")

    # 1. Validate file extension
    is_img = is_image_file(filename)
    is_vid = is_video_file(filename)

    if not is_img and not is_vid:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload an image (JPG, PNG, etc.) or video (MP4, AVI, MOV, etc.)."
        )

    # 2. Save file temporarily
    try:
        temp_path = save_uploaded_file(file, TEMP_DIR)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store uploaded file: {e}")

    # Register temporary file cleanup to guarantee deletion
    if background_tasks:
        background_tasks.add_task(clean_up_file, temp_path)

    results: List[TimestampDetection] = []
    total_crops = 0
    categories_found: Set[str] = set()
    media_type = "image" if is_img else "video"
    duration_seconds = None

    try:
        if is_img:
            # --- PROCESS IMAGE ---
            frame = cv2.imread(str(temp_path))
            if frame is None:
                raise HTTPException(status_code=400, detail="Could not decode the uploaded image.")

            detections = detector.detect_and_crop(frame, timestamp=0.0)
            
            if detections:
                results.append(TimestampDetection(
                    timestamp_seconds=0.0,
                    detections=[DetectionDetail(**d) for d in detections]
                ))
                total_crops = len(detections)
                categories_found.update(d["class_name"] for d in detections)

        else:
            # --- PROCESS VIDEO ---
            cap = cv2.VideoCapture(str(temp_path))
            if not cap.isOpened():
                raise HTTPException(status_code=400, detail="Could not open the uploaded video.")

            # Get video specs
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Prevent DivisionByZero if metadata is corrupt
            if fps <= 0:
                fps = 30.0
            
            duration_seconds = round(frame_count / fps, 2)
            logger.info(f"Video specs: {duration_seconds}s duration, {fps} FPS, {frame_count} frames.")

            # Calculate frame sampling step
            # E.g., if video is 30 FPS and we want 2 FPS sampling, we take a frame every 15 frames.
            step = int(round(fps / VIDEO_FPS_SAMPLING))
            step = max(1, step)  # Enforce at least 1

            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % step == 0:
                    timestamp = round(frame_idx / fps, 2)
                    detections = detector.detect_and_crop(frame, timestamp=timestamp)
                    
                    if detections:
                        results.append(TimestampDetection(
                            timestamp_seconds=timestamp,
                            detections=[DetectionDetail(**d) for d in detections]
                        ))
                        total_crops += len(detections)
                        categories_found.update(d["class_name"] for d in detections)

                frame_idx += 1

            cap.release()

    except Exception as e:
        logger.error(f"Error during video/image analysis: {e}")
        # Clean up temp file immediately on exception
        clean_up_file(temp_path)
        raise HTTPException(status_code=500, detail=f"Inference error during processing: {e}")

    finally:
        # Guarantee cleanup if background tasks are not supported or registered
        if not background_tasks:
            clean_up_file(temp_path)

    return AnalysisResponse(
        filename=filename,
        media_type=media_type,
        duration_seconds=duration_seconds,
        total_detections_count=total_crops,
        categories_found=list(categories_found),
        results=results
    )
