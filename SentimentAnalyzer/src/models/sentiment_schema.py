# src/models/sentiment_schema.py
from __future__ import annotations
from typing import List, Literal
from pydantic import BaseModel, Field, field_validator

class SentimentResult(BaseModel):
    company_name: str = Field(..., description="Canonical name of the company being analyzed.")
    stock_code: str = Field(..., description="Resolved stock ticker/symbol.")
    newsdesc: str = Field(..., description="Short bullet list of recent headlines with links.")
    sentiment: Literal["Positive", "Negative", "Neutral"] = Field(..., description="Overall market sentiment.")
    people_names: List[str] = Field(default_factory=list, description="People referenced in the news.")
    places_names: List[str] = Field(default_factory=list, description="Places referenced.")
    other_companies_referred: List[str] = Field(default_factory=list, description="Other companies mentioned.")
    related_industries: List[str] = Field(default_factory=list, description="Related industries/sectors.")
    market_implications: str = Field("", description="Short analyst-style implications for the market.")
    confidence_score: float = Field(0.5, description="0.0–1.0 confidence in the sentiment assessment.")

    @field_validator("confidence_score")
    @classmethod
    def _clamp_confidence(cls, v: float) -> float:
        try:
            return max(0.0, min(1.0, float(v)))
        except Exception:
            return 0.5

    model_config = {
        "extra": "forbid",   # reject unexpected keys
        "validate_assignment": True
    }
