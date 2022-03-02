from typing import List, Dict, Any

from pydantic import BaseModel, Field


class Prediction(BaseModel):
    """Represents predicted fields of study for a given paper"""

    # foses: List[str] = Field(description="Predicted fields of study")
    scores: List[Dict[str, Any]] = Field(
        description="Descision scores for all fields of study"
    )
