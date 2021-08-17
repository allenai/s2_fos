from typing import List

from pydantic import BaseModel, Field


class Prediction(BaseModel):
    """Represents predicted fields of study for a given paper"""

    foses: List[str] = Field(description="Predicted fields of study")
