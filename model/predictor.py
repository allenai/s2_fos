from model.multioutput import MultiOutputClassifierWithDecision
from model.prediction import Prediction
from model.decisionscore import DecisionScore
from model.instance import Instance
import logging
import os
import numpy as np
from typing import List, Optional

from pydantic import BaseModel, BaseSettings, Field
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from model import utils
from model.hyperparameters import ModelHyperparameters


logger = logging.getLogger(__name__)


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

    _feature_pipe: Pipeline
    _mlb: MultiLabelBinarizer
    _classifier: MultiOutputClassifierWithDecision

    def __init__(self, config: PredictorConfig):
        self._classifier, self._fasttext = utils.load_model(
            config.model_artifacts_dir()
        )
        self._feature_pipe, self._mlb = utils.load_feature_pipe(
            config.model_artifacts_dir()
        )

        # ensure that cached_dict has been built on mlb
        self._mlb._cached_dict = dict(
            zip(self._mlb.classes_, range(len(self._mlb.classes_)))
        )
        self.mlb_inverse_dict = {v: k for k, v in self._mlb._cached_dict.items()}

    # works for single text at a time
    def get_decision_scores(self, original_text) -> List[DecisionScore]:

        # Skip the prediction process if the input text is not in english
        is_english = utils.detect_language(self._fasttext, original_text)[1]
        if not is_english:
            return []

        # featurize the original text
        featurized_text = self._feature_pipe.transform([original_text])[0]
        decision_scores = self._classifier.decision_function(featurized_text)

        # mlb_inverse_dict: {0: 'Agricultural and Food sciences', 1: 'Art', 2: 'Biology',... 21: 'Psychology', 22: 'Sociology'}

        model_predictions: List[DecisionScore] = [
            DecisionScore(label=label, score=decision_scores[int(index)])
            for index, label in self.mlb_inverse_dict.items()
        ]

        return model_predictions

    def predict_batch(self, instances: List[Instance]) -> List[Prediction]:
        texts = [utils.make_inference_text(instance) for instance in instances]

        return [Prediction(scores=self.get_decision_scores(text)) for text in texts]
