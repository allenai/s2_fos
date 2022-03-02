from typing import List

from model.decisionscore import DecisionScore
from pydantic import BaseModel, Field


class Prediction(BaseModel):
    """Represents predicted scores for all fields of study for a given paper"""

    scores: List[DecisionScore] = Field(
        description="Descision scores for all fields of study"
    )
