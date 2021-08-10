from pydantic import BaseModel, Field


class Instance(BaseModel):
    """Represents one object for which inference can be performed."""

    field1: str = Field(description="Some string field of consequence for inference")
    field2: float = Field(description="Some float field of consequence for inference")
