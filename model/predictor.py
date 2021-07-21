import json
import os
from typing import List

from pydantic import BaseModel, BaseSettings, Field


class ModelConfig(BaseSettings):
    """
    The set of configuration parameters required to instantiate a model
    and initialize it for inference over the lifetime of the process.
    """

    artifacts_dir: str = Field(
        description="Directory to find model artifacts such as learned parameters, etc",
    )


class Instance(BaseModel):
    """Represents one object for which inference can be performed."""

    field1: str = Field(description="Some string field of consequence for inference")
    field2: float = Field(description="Some float field of consequence for inference")


class Prediction(BaseModel):
    """Represents the result of inference over one instance"""

    output_field: str = Field(description="Some predicted piece of data")


class Predictor:
    _cool_learned_factor: int
    _config: ModelConfig

    def __init__(self, config: ModelConfig):
        """
        Initialize your model using the passed parameters
        """
        self._config = config
        self._load_learned_parameters()

    def _load_learned_parameters(self):
        params_path = os.path.join(
            self._config.artifacts_dir, "example_learned_parameters.json"
        )

        with open(params_path, "r") as f:
            as_dict = json.loads(f.read())
            self._cool_learned_factor = as_dict["cool_learned_factor"]

    def predict_batch(self, instances: List[Instance]) -> List[Prediction]:
        predictions = []

        for instance in instances:
            better_field2 = instance.field2 * self._cool_learned_factor
            predictions.append(
                Prediction(output_field=f"{instance.field1}:{better_field2}")
            )

        return predictions
