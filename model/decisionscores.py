from typing import List, Dict, Any

from pydantic import BaseModel, Field


class DecisionScore(BaseModel):
    label: str
    score: float


class DecisionScores(BaseModel):
    """Represents decision scores predicted for fields of study for a given paper"""

    scores: List[Dict[str, Any]] = Field(
        description="Descision scores for all fields of study"
    )

    # [{label=Agricultural And Food Sciences, score=-1.431}, {label=Art, score=-1.53}, {label=Biology, score=-1.997}, {label=Business, score=-0.428}, {label=Chemistry, score=-1.159}, {label=Computer Science, score=-1.505}, {label=Economics, score=0.699}, {label=Education, score=-1.338}, {label=Engineering, score=-1.461}, {label=Environmental Science, score=-1.52}, {label=Geography, score=-1.417}, {label=Geology, score=-1.54}, {label=History, score=0.041}, {label=Law, score=-0.976}, {label=Linguistics, score=-1.604}, {label=Materials Science, score=-0.977}, {label=Mathematics, score=-1.425}, {label=Medicine, score=-1.083}, {label=Philosophy, score=-1.294}, {label=Physics, score=-1.268}, {label=Political Science, score=-0.772}, {label=Psychology, score=-1.175}, {label=Sociology, score=-0.614}]
