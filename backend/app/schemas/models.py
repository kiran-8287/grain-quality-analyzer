"""
Pydantic Schemas for Rice Quality Analysis API.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    service: str = "GRAIN QUALITY ANALYZER Backend"
    gpu_available: bool = False
    device: str = "cpu"


class ModelRegistryItem(BaseModel):
    name: str
    status: str
    version: str
    type: str
    weights_path: str
    validation_metrics: Optional[Dict[str, Any]] = None


class ModelsResponse(BaseModel):
    models: Dict[str, Any]


class StandardsResponse(BaseModel):
    standard_name: str
    source_title: str
    source_url: str
    issue_date: str
    season: str
    commodity: str
    grade_a: Dict[str, Any]
    common: Dict[str, Any]
    footnotes: List[str]
    image_assessment_mapping: Dict[str, Any]


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # 'pending', 'processing', 'completed', 'failed'
    progress_stage: str
    progress_percent: int
    created_at: float
    completed_at: Optional[float] = None
    error: Optional[str] = None


class DefectDetail(BaseModel):
    label: Optional[str] = None
    status: Optional[str] = None
    probability: Optional[float] = None
    confidence: Optional[float] = None
    method: str
    limitation: Optional[str] = None
    source_type: Optional[str] = None


class GrainGeometryModel(BaseModel):
    grain_id: int
    area_pixels: float
    perimeter_pixels: float
    centroid: List[float]
    orientation_deg: float
    bbox: List[int]
    major_axis_pixels: float
    minor_axis_pixels: float
    solidity: float
    aspect_ratio: float
    length_pixels: float
    breadth_pixels: float
    length_mm: Optional[float] = None
    breadth_mm: Optional[float] = None
    lb_ratio: Optional[float] = None
    measurement_quality: str


class GrainDetailModel(BaseModel):
    id: int
    bbox: List[int]
    centroid: List[float]
    confidence: float
    segmentation_quality: str
    is_touching: bool
    contour: List[List[int]]
    geometry: Dict[str, Any]
    defects: Dict[str, Any]
    sample_level_parameters: Dict[str, str]


class AnalysisResultResponse(BaseModel):
    success: bool
    rice_detected: bool
    message: Optional[str] = None
    image: Optional[Dict[str, Any]] = None
    sample: Optional[Dict[str, Any]] = None
    calibration: Optional[Dict[str, Any]] = None
    quality: Optional[Dict[str, Any]] = None
    grains: List[Dict[str, Any]] = Field(default_factory=list)
    foreign_matter: List[Dict[str, Any]] = Field(default_factory=list)
    admixture: Optional[Dict[str, Any]] = None
    summary: Optional[Dict[str, Any]] = None
    standards: Optional[Dict[str, Any]] = None
    annotated_image_base64: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    processing_time_seconds: Optional[float] = None


class ExperimentalGrainInstance(BaseModel):
    id: int
    class_name: str = Field(alias="class")
    score: float
    length_px: float
    breadth_px: float
    lb_ratio: Optional[float] = None
    length_mm: Optional[float] = None
    breadth_mm: Optional[float] = None
    bbox: List[int]
    area_px: float
    orientation_deg: float
    measurable: bool
    flags: List[str] = Field(default_factory=list)


class ExperimentalForeignDetection(BaseModel):
    class_name: str = Field(alias="class")
    score: float
    bbox: List[float]


class ExperimentalGrainAnalysisResponse(BaseModel):
    experimental: Literal[True] = True
    image_size: Dict[str, int]
    counts: Dict[str, int]
    percentages: Dict[str, Optional[float]]
    grains: List[ExperimentalGrainInstance] = Field(default_factory=list)
    foreign_matter: List[ExperimentalForeignDetection] = Field(default_factory=list)
    aggregates: Dict[str, Any]
    model_versions: Dict[str, Any]
    device: str
    timings_ms: Dict[str, float]
