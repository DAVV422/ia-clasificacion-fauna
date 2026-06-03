import os
import sys
from pathlib import Path

# Base paths
if getattr(sys, 'frozen', False):
    # Si estamos en un .exe de PyInstaller, BASE_DIR es la carpeta donde está el .exe
    BASE_DIR = Path(sys.executable).parent
else:
    # Si es script normal
    BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"
DETECTIONS_DIR = BASE_DIR / "detections"
TEMP_DIR = BASE_DIR / "temp"

# Ensure crucial directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# API Configurations
API_TITLE = "WWF Wildlife Detection API (MegaDetector)"
API_VERSION = "1.0.0"

# Model configuration
MEGADETECTOR_URL = "https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt"
MEGADETECTOR_PATH = MODELS_DIR / "md_v5a.0.0.pt"
CONFIDENCE_THRESHOLD = 0.30

# SpeciesNet configuration
# Identifier for Google's SpeciesNet classifier model (hosted on Kaggle, automatically downloaded via kagglehub)
SPECIESNET_MODEL_NAME = "kaggle:google/speciesnet/pyTorch/v4.0.2a/1"
SPECIESNET_CONFIDENCE_THRESHOLD = 0.15


# Video sampling configuration
# How many frames per second of video to process (e.g. 2.0 = process 2 frames for every 1 second of video)
# Higher means more precision, lower means faster processing
VIDEO_FPS_SAMPLING = 2.0

# Class mappings for MegaDetector v5a/v5b
# MegaDetector output categories: 1 -> animal, 2 -> person, 3 -> vehicle
# Note: For some YOLO configurations they are 0 -> animal, 1 -> person, 2 -> vehicle.
# MegaDetector official class map:
# 1 = animal
# 2 = person
# 3 = vehicle
# Let's map both index-based and float/int outputs carefully.
MEGADETECTOR_CLASSES = {
    "1": "animal",
    "2": "person",
    "3": "vehicle",
    "0": "animal" # fallback for 0-indexed YOLO conversions
}

# Static path prefix for crops
DETECTIONS_URL_PREFIX = "/detections"
