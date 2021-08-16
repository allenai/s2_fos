import logging
import os
from typing import List, Optional

from pydantic import BaseModel, BaseSettings, Field
from sklearn.multioutput import MultiOutputClassifier

from model import utils
from model.hyperparameters import ModelHyperparameters


logger = logging.getLogger(__name__)

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
    model_version: Optional[str] = Field(
        description="Logical name for a model version or experiment, to segment and retrieve artifacts",
        default=None,
    )

    def model_artifacts_dir(self) -> str:
        if self.model_version is None or self.model_version == "":
            return self.artifacts_dir

        return os.path.join(self.artifacts_dir, self.model_version)


class Predictor:
    """
    Loads in a trained classifier and TFIDF vectorizer.
    Used to produce batches of classification predictions.
    """

    _hyperparameters: ModelHyperparameters
    _classifier: MultiOutputClassifier

    def __init__(self, config: PredictorConfig):
        self._hyperparameters, self._classifier = utils.load_model(
            config.model_artifacts_dir()
        )

    def predict_batch(self, instances: List[Instance]) -> List[Prediction]:
        texts = [
            utils.make_inference_text(instance, self._hyperparameters.use_abstract)
            for instance in instances
        ]

        multihot_preds = self._classifier.predict(texts)

        return [
            Prediction(foses=utils.multihot_to_labels(multihot))
            for multihot in multihot_preds
        ]
