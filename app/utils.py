import logging
import shutil
import uuid
from pathlib import Path
import requests
from fastapi import UploadFile

logger = logging.getLogger("uvicorn.error")

def download_megadetector_weights(url: str, dest_path: Path):
    """
    Downloads MegaDetector v5a model weights from URL if not already present.
    Uses chunked downloading to prevent high memory usage.
    """
    if dest_path.exists():
        logger.info(f"MegaDetector model found at {dest_path}. Skipping download.")
        return

    logger.info(f"MegaDetector weights not found. Downloading from {url}...")
    logger.info("This download (~144MB) happens only once on startup and may take a moment...")
    
    # Ensure directory exists
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    temp_dest = dest_path.with_suffix(".tmp")
    try:
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            downloaded = 0
            
            with open(temp_dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            logger.info(f"Download progress: {percent:.1f}% ({downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB)")
            
        temp_dest.rename(dest_path)
        logger.info("MegaDetector weights successfully downloaded!")
    except Exception as e:
        if temp_dest.exists():
            temp_dest.unlink()
        logger.error(f"Failed to download MegaDetector weights: {e}")
        raise RuntimeError(f"Could not initialize MegaDetector model: {e}")

def save_uploaded_file(upload_file: UploadFile, temp_dir: Path) -> Path:
    """
    Saves a FastAPI UploadFile temporarily on local storage and returns its path.
    Generates a unique name to avoid conflicts.
    """
    temp_dir.mkdir(parents=True, exist_ok=True)
    file_extension = Path(upload_file.filename).suffix
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    target_path = temp_dir / unique_filename

    try:
        with target_path.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return target_path
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        if target_path.exists():
            target_path.unlink()
        raise e

def clean_up_file(file_path: Path):
    """
    Safely deletes a file if it exists.
    """
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        logger.error(f"Failed to delete temporary file {file_path}: {e}")
