from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, Field

api = FastAPI()


class Instance(BaseModel):
    """Represents one object for which inference can be performed."""
    field1: str = Field(description="Some string field of consequence for inference")
    field2: float = Field(description="Some float field of consequence for inference")


class Prediction(BaseModel):
    """Represents the result of inference over one instance"""
    classification: str = Field(description="Some predicted class in this simple example")


class InvocationsRequest(BaseModel):
    """Represents the JSON body of a set of inference requests."""
    instances: List[Instance] = Field(description="A list of Instances over which to perform inference")


class InvocationsResponse(BaseModel):
    """The results of inference over each passed instance"""
    predictions: List[Prediction] = Field(description="The predictions")


@api.post("/invocations", )
async def invocations(req: InvocationsRequest) -> InvocationsResponse:
    """Accepts JSON or JSONL application types, returning the same MIME types"""

    resp = InvocationsResponse(
        predictions=[
            Prediction(f"{inst.field1}:{inst.field2}")
            for inst in req.instances
        ]
    )

    return resp


@api.get("/ping")
async def health_check():
    return {"message": "Okalee-dokalee"}
