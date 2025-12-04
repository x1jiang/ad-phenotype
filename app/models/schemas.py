"""
Pydantic schemas for request/response models
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class AnalysisType(str, Enum):
    """Analysis type enumeration"""
    LOW_DIM = "low_dim"
    ASSOCIATION = "association"
    NETWORK = "network"


class StratificationType(str, Enum):
    """Stratification type"""
    NONE = "none"
    SEX = "sex"
    AGE = "age"


class ControlMatchingRequest(BaseModel):
    """Request model for control matching"""
    matching_vars: List[str] = Field(
        default=["Race", "Age", "Sex", "Death_Status"],
        description="Variables to match on"
    )
    ratio: int = Field(default=2, description="Control to case ratio")
    sensitivity: bool = Field(
        default=False,
        description="Include encounter count and duration in matching"
    )


class UMAPRequest(BaseModel):
    """Request model for UMAP analysis"""
    n_neighbors: int = Field(default=15, ge=2, le=200)
    min_dist: float = Field(default=0.1, ge=0.0, le=1.0)
    metric: str = Field(default="cosine")
    random_state: int = Field(default=42)


class AssociationRequest(BaseModel):
    """Request model for association analysis"""
    analysis_type: str = Field(
        default="diagnosis",
        description="Type: diagnosis, medication, or lab"
    )
    stratify_by: Optional[str] = Field(
        default=None,
        description="Stratify by: sex, age, or None"
    )
    bonferroni_alpha: float = Field(default=0.05)


class NetworkRequest(BaseModel):
    """Request model for network analysis"""
    cutoff: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Minimum proportion of patients for edge inclusion"
    )
    diagnosis_level: str = Field(
        default="DiagnosisName",
        description="Diagnosis level: DiagnosisName, Level3_Category, or Level2_Category"
    )


class AnalysisResponse(BaseModel):
    """Generic analysis response"""
    status: str
    message: str
    data: Optional[Dict[str, Any]] = None
    job_id: Optional[str] = None


class PlotResponse(BaseModel):
    """Plot response"""
    plot_type: str
    data: Dict[str, Any]
    format: str = "json"

