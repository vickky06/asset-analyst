from pydantic import BaseModel, Field, field_validator
from typing import  List, Dict, Any
import json
class InvestmentAnalysis(BaseModel):
    thesis: str = Field(description="Investment thesis and reasoning")
    risks: str = Field(description="Key risks and concerns")
    valuation_view: str = Field(description="Valuation perspective and metrics")
    catalysts: str = Field(description="Potential catalysts and events")
    confidence: float = Field(description="Confidence score 0-1", ge=0, le=1)
    score: int = Field(description="Investment score 0-100", ge=0, le=100)
    recommendation: str = Field(description="Investment recommendation", pattern="^(BUY|HOLD|SELL)$")
    assumptions: str = Field(description="Key assumptions made")
    time_horizon: str = Field(description="Investment time horizon")

class ResearchData(BaseModel):
    sources: List[Dict[str, str]] = Field(description="List of sources with title and url")
    summary: str = Field(description="Research summary")
    key_metrics: Dict[str, Any] = Field(description="Key financial or market metrics")

    @field_validator('key_metrics', mode='before')
    @classmethod
    def parse_key_metrics(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                # If it's not valid JSON, treat it as a single key-value pair
                return {"raw_metrics": v}
        return v
    
    @field_validator('sources', mode='before')
    @classmethod
    def parse_sources(cls, v):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return []
        return v