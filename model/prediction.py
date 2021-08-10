from pydantic import BaseModel, Field


class Prediction(BaseModel):
    """Represents the result of inference over one instance"""

    output_field: str = Field(description="Some predicted piece of data")
