from typing import Optional

from pydantic import BaseModel, Field


class Instance(BaseModel):
    """Represents one paper for which we can predict fields of study"""

    title: str = Field(description="Title text for paper")
    abstract: Optional[str] = Field(description="Abstract text for paper (optional)")
