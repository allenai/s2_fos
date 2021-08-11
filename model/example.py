from pydantic import BaseModel

from model.instance import Instance
from model.prediction import Prediction


class Example(BaseModel):
    instance: Instance
    labels: Prediction
