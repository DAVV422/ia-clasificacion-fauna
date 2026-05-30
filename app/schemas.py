from pydantic import BaseModel, Field
from typing import List, Optional

class DetectionDetail(BaseModel):
    box: List[float] = Field(..., description="Bounding box in normalized coordinates [ymin, xmin, ymax, xmax]")
    class_name: str = Field(..., description="The detected category name (animal, person, vehicle)")
    confidence: float = Field(..., description="Inference confidence score (0.0 to 1.0)")
    species: Optional[str] = Field(None, description="The classified animal species name (e.g. Panthera onca)")
    species_confidence: Optional[float] = Field(None, description="Species classification confidence score")
    crop_path: Optional[str] = Field(None, description="Local relative path where the crop is saved")
    crop_url: Optional[str] = Field(None, description="Static HTTP URL to access the crop image")


class TimestampDetection(BaseModel):
    timestamp_seconds: float = Field(..., description="Timestamp in seconds where the detection occurred")
    detections: List[DetectionDetail] = Field(..., description="List of detections in this frame/timestamp")

class AnalysisResponse(BaseModel):
    filename: str = Field(..., description="Name of the processed file")
    media_type: str = Field(..., description="Type of media processed: image or video")
    duration_seconds: Optional[float] = Field(None, description="Duration in seconds (only for video)")
    total_detections_count: int = Field(..., description="Total number of cropped detections generated")
    categories_found: List[str] = Field(..., description="List of unique categories found in the file")
    results: List[TimestampDetection] = Field(..., description="Chronological detections ordered by timestamp")
