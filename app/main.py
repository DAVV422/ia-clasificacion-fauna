import logging
from pathlib import Path
from typing import List, Set
import cv2
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

            detections = detector.detect_and_crop(frame, filename, timestamp=0.0)
            
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

            # =========================================================================
            # 🛠️ NUEVA LÓGICA DE MUESTREO: Mínimo 1, Máximo 3 capturas distribuidas uniformemente
            # =========================================================================
            MAX_CAPTURES = 3 
            # Aseguramos que el límite configurado esté entre 1 y 3 de manera segura
            total_captures_to_make = max(1, min(3, MAX_CAPTURES)) 
            
            # Calculamos los índices exactos de los frames que vamos a extraer
            frames_to_extract = []
            if frame_count > 0:
                interval = frame_count / (total_captures_to_make + 1)
                for i in range(1, total_captures_to_make + 1):
                    frames_to_extract.append(int(interval * i))
            else:
                # Si por algún motivo el frame_count viene en 0, procesamos al menos el primer frame
                frames_to_extract = [0]

            # Inicializamos la métrica solicitada para contar animales en simultáneo
            max_animals_simultaneous = 0      

            # 1. Obtener el nombre base original
            base_name_raw = Path(filename).stem

            # 2. Normalizar para eliminar acentos (Ej: "jaguár" -> "jaguar", "niño" -> "nino")
            base_name_clean = unicodedata.normalize('NFKD', base_name_raw).encode('ASCII', 'ignore').decode('ASCII')

            # 3. Reemplazar espacios por guiones bajos (_)
            base_name_clean = base_name_clean.replace(" ", "_")

            # 4. Remover cualquier caracter que NO sea una letra, número, guion medio o guion bajo
            base_name_clean = re.sub(r'[^a-zA-Z0-9_ -]', '', base_name_clean)

            # 5. Opcional: Convertir todo a minúsculas para estandarizar las rutas
            base_name_clean = base_name_clean.lower()

            # 6. Generar el ID único y la carpeta final
            unique_run_id = uuid.uuid4().hex[:8]
            video_execution_folder = f"{base_name_clean}_{unique_run_id}"
            # =========================================================================

            # Iteramos directo sobre los frames seleccionados
            for frame_idx in frames_to_extract:
                # Posicionamos el puntero de OpenCV directamente en el frame deseado (Evita el lag del while)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                timestamp = round(frame_idx / fps, 2)
                
                # Pasamos el identificador del video único a tu función modificada
                detections = detector.detect_and_crop(frame, video_execution_folder, timestamp=timestamp)
                if detections:
                # 📊 CALCULAR EL MÁXIMO DE ANIMALES EN SIMULTÁNEO EN ESTE FRAME
                    animals_in_frame = sum(1 for d in detections if d["class_name"] == "animal")
                    if animals_in_frame > max_animals_simultaneous:
                        max_animals_simultaneous = animals_in_frame

                    # 🛠️ APLANAR Y ASIGNAR EL CROP URL A CADA DETECTION DETAIL
                    flattened_detections = []

                    for d in detections:
                        # Obtenemos la lista de especies devueltas por la función
                        species_list = d.get("species", [])
                        
                        if species_list:
                            # La primera es la que tiene mayor confianza
                            top_species = species_list[0]
                            s_name = top_species["species"]
                            s_conf = top_species["confidence"]
                        else:
                            # Fallback para no-animales (vehículos, personas) o si SpeciesNet falló
                            s_name = d.get("class_name", "unknown")
                            s_conf = float(d.get("confidence", 0.0))

                        # Creamos el detalle inyectando su respectivo crop_url
                        flattened_detections.append(DetectionDetail(
                            species=s_name,
                            confidence=round(s_conf, 2),  # Redondeado a 2 decimales (ej: 0.95)
                            crop_url=d.get("crop_url")    # 📂 Asignado individualmente
                        ))

                    # Agregamos el frame con su lista de detecciones detalladas
                    results.append(TimestampDetection(
                        timestamp_seconds=timestamp,
                        detections=flattened_detections
                    ))
                    
                    total_crops += len(detections)
                    categories_found.update(d["class_name"] for d in detections)

            # Liberamos el lector para evitar bloqueos del SO (WinError 32)
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
