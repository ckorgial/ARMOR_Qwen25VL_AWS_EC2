from pydantic import BaseModel, Field, field_serializer
from typing import List, Literal, Optional, Any, Dict
from datetime import datetime

class AOIEntry(BaseModel):
    area: str
    reasoning: str
    conclusion: str
    isAI: Literal["true", "false"]

class InfoAgentReport(BaseModel):
    agent: str = "InfoAgent"
    image_id: str
    model: str
    created_at: datetime
    areas: List[str]
    results: List[AOIEntry]
    raw_text: Optional[str] = None
    meta: Optional[Any] = None

    @field_serializer("created_at")
    def _ser_created_at(self, v: datetime):
        return v.isoformat()

class ConductorObjective(BaseModel):
    image_id: str
    target_class: int = 0
    ssim_min: float = 0.70
    conf_target: float = 0.90
    epsilon_max: float = 12/255
    aggregate: Literal["mean", "sum"] = "mean"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    note: Optional[str] = None

    @field_serializer("created_at")
    def _ser_created_at(self, v: datetime):
        return v.isoformat()

class TrackMetrics(BaseModel):
    ssim: Optional[float] = None
    conf_resnet50: Optional[float] = None
    conf_densenet121: Optional[float] = None
    avg_conf: Optional[float] = None

class TrackMeta(BaseModel):
    agent: str
    method: str
    params: Dict[str, Any] = {}
    metrics: TrackMetrics = Field(default_factory=TrackMetrics)
    image_path: str
    delta_path: Optional[str] = None
    step: int = 0
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @field_serializer("updated_at")
    def _ser_updated_at(self, v: datetime):
        return v.isoformat()

class MasterMeta(BaseModel):
    weights: Dict[str, float]
    metrics: TrackMetrics
    image_path: str
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @field_serializer("updated_at")
    def _ser_updated_at(self, v: datetime):
        return v.isoformat()

class CritiqueEntry(BaseModel):
    name: str
    kind: Literal["surrogate", "perceptual"]
    metrics: Dict[str, Any]
    source: str
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @field_serializer("updated_at")
    def _ser_updated_at(self, v: datetime):
        return v.isoformat()

class FeedbackPanel(BaseModel):
    image_id: str
    entries: List[CritiqueEntry] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @field_serializer("updated_at")
    def _ser_updated_at(self, v: datetime):
        return v.isoformat()

# New: strategy
class StrategyDoc(BaseModel):
    image_id: str
    suggestions: Dict[str, Dict[str, Any]] = Field(default_factory=dict)  # per-method
    mixer_weights: Dict[str, float] = Field(default_factory=dict)
    rationale: Optional[str] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @field_serializer("updated_at")
    def _ser_updated_at(self, v: datetime):
        return v.isoformat()