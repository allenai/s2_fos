import json
import os
from typing import List

from pydantic import BaseModel, BaseSettings, Field

from model.instance import Instance
from model.prediction import Prediction


class PredictorConfig(BaseSettings):
    """
    The set of configuration parameters required to instantiate a predictor and
    initialize it for inference. This is an appropriate place to specify any parameter
    or configuration values your model requires that aren't packaged with your
    versioned model artifacts. These should be rare beyond the included
    `artifacts_dir`.

    Values for these config fields can be provided as environment variables, see:
    `./docker.env`
    """

    artifacts_dir: str = Field(
        description="Directory to find model artifacts such as learned model parameters, etc",
        default="/opt/ml/model",
    )


class Predictor:
    """
    Used by the included FastAPI server to perform inference. Initialize your model
    in the constructor using the supplied `PredictorConfig` instance, and perform inference
    for each `Instance` passed via `predict_batch()`. The default batch size is `1`, but
    you should handle as many `Instance`s as are provided.
    """

    _cool_learned_factor: int
    _config: PredictorConfig

    def __init__(self, config: PredictorConfig):
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
