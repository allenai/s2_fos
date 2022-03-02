from pydantic import BaseModel, Field


class DecisionScore(BaseModel):
    """Represents decision score predicted for a given field of study for a given paper"""

    label: str
    score: float
