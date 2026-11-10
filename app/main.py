import logging
from pathlib import Path
from typing import List, Set
import re
import unicodedata
import uuid
from pathlib import Path
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
    DETECTIONS_URL_PREFIX,
)
from app.utils import download_megadetector_weights, save_uploaded_file, clean_up_file
from app.detector import WildlifeDetector
from app.schemas import AnalysisResponse, TimestampDetection, DetectionDetail
from wwm_pipeline.config import PipelineConfig
from wwm_pipeline.errors import PipelineError
from wwm_pipeline.pipeline import classify_file

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


def _execution_folder(filename: str) -> str:
    """Carpeta única de los recortes de un video (igual que antes): nombre normalizado + 8 hex."""
    base_name_clean = unicodedata.normalize('NFKD', Path(filename).stem).encode('ASCII', 'ignore').decode('ASCII')
    base_name_clean = base_name_clean.replace(" ", "_")
    base_name_clean = re.sub(r'[^a-zA-Z0-9_ -]', '', base_name_clean).lower()
    return f"{base_name_clean}_{uuid.uuid4().hex[:8]}"


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
        # HU-IA-40: la clasificación vive en wwm_pipeline (preset legacy_api = comportamiento actual).
        if is_img:
            run_name = filename
        else:
            run_name = _execution_folder(filename)
        config = PipelineConfig.legacy_api().with_crops(DETECTIONS_DIR, DETECTIONS_URL_PREFIX)
        try:
            outcome = classify_file(temp_path, filename, config, detector.models, run_name=run_name)
        except PipelineError as perr:
            # Mismo mensaje que antes; como antes, el except general lo convierte en 500.
            detail = "Could not decode the uploaded image." if is_img else "Could not open the uploaded video."
            raise HTTPException(status_code=400, detail=detail) from perr

        if is_img:
            # --- PROCESS IMAGE ---
            detections = outcome.frames[0].detections
            if detections:
                # ⚠️ Error conocido (IA-03): DetectionDetail(**d) falla porque "species" es una lista.
                results.append(TimestampDetection(
                    timestamp_seconds=0.0,
                    detections=[DetectionDetail(**d) for d in detections]
                ))
                total_crops = len(detections)
                categories_found.update(d["class_name"] for d in detections)
        else:
            # --- PROCESS VIDEO ---
            duration_seconds = outcome.duration_seconds
            logger.info(f"Video specs: {duration_seconds}s duration, {outcome.fps} FPS, {outcome.frame_count} frames.")
            for frame_result in outcome.frames:
                detections = frame_result.detections
                if not detections:
                    continue
                flattened_detections = []
                for d in detections:
                    species_list = d.get("species", [])
                    if species_list:
                        top_species = species_list[0]
                        s_name = top_species["species"]
                        s_conf = top_species["confidence"]
                    else:
                        s_name = d.get("class_name", "unknown")
                        s_conf = float(d.get("confidence", 0.0))
                    flattened_detections.append(DetectionDetail(
                        species=s_name,
                        confidence=round(s_conf, 2),
                        crop_url=d.get("crop_url")
                    ))
                results.append(TimestampDetection(
                    timestamp_seconds=frame_result.timestamp_seconds,
                    detections=flattened_detections
                ))
                total_crops += len(detections)
                categories_found.update(d["class_name"] for d in detections)

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
