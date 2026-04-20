from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class ReportRequest(BaseModel):
    suspect_drugs:    str   = Field(..., description="Pipe-separated suspect drug names")
    reactions:        str   = Field(..., description="Pipe-separated MedDRA reaction terms")
    outcome_severity: int   = Field(1, ge=1, le=5, description="1=Other → 5=Death")
    age_yr:           Optional[float] = Field(None, ge=0, le=120)
    wt_kg:            Optional[float] = Field(None, ge=1, le=300)
    sex_enc:          Optional[float] = Field(None, ge=0, le=1)

    model_config = {"json_schema_extra": {"example": {
        "suspect_drugs": "WARFARIN|ASPIRIN",
        "reactions": "HAEMORRHAGE|DYSPNOEA",
        "outcome_severity": 4,
        "age_yr": 72.0,
        "wt_kg": 68.0,
        "sex_enc": 0,
    }}}


class SignalResponse(BaseModel):
    anomaly_score: float
    flagged:       bool
    risk_tier:     Literal["HIGH", "MEDIUM", "LOW"]


class BatchRequest(BaseModel):
    reports: List[ReportRequest] = Field(..., max_length=500)


class BatchResponse(BaseModel):
    total_scored:  int
    total_flagged: int
    results:       List[SignalResponse]


class HealthResponse(BaseModel):
    status:       str
    model_type:   str
    best_config:  str
    version:      str
