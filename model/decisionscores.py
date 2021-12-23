from typing import List

from pydantic import BaseModel, Field


class DecisionScores(BaseModel):
    """Represents decision scores predicted for fields of study for a given paper"""

    scores: dict = Field(description="Decision scores for all fields of study")
