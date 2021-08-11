import logging
import os
from typing import List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, BaseSettings, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC

from model.hyperparameters import ModelHyperparameters
from model.example import Example
from model.utils import labels_to_multihot, make_inference_text


logger = logging.getLogger(__name__)
RANDOM_SEED = 1337  # as is tradition


class TrainingConfig(BaseSettings):
    """
    The set of configuration parameters required to run a training routine.

    Values for these config fields can be provided as environment variables, see:
    `./docker.env`
    """

    artifacts_dir: str = Field(
        description="Directory to find model artifacts such as learned model parameters, etc",
        default="/opt/ml/model",
    )
    input_data_dir: str = Field(
        description="Directory to find data files",
        default="/opt/ml/input/_input_data"
    )
    channel_name: str = Field(
        description="Name of data channel to use for training. Name of subdirectory under `input_data_dir`",
        default="training"
    )
    input_config_dir: str = Field(
        description="Directory to find configuration files like hyperparameters",
        default="/opt/ml/input/input_config"
    )
    hyperparameters_file: str = Field(
        description="Filename of hyperparameters json file, following specification in `model.hyperparameters`"
    )
    model_version: Optional[str] = Field(
        description="Logical name for a model version or experiment, to segment and retrieve artifacts",
        default=None
    )

    def training_data_dir(self) -> str:
        return os.path.join(self.input_data_dir, self.channel_name)

    def target_artifacts_dir(self) -> str:
        if self.model_version is None:
            return self.artifacts_dir

        return os.path.join(self.artifacts_dir, self.model_version)

    def load_hyperparameters(self) -> ModelHyperparameters:
        filepath = os.path.join(self.hyperparameters_file, self.hyperparameters_file)
        return ModelHyperparameters.parse_file(filepath)


def build_and_train_model(
    training_examples: List[Example], hyperparameters: ModelHyperparameters
) -> Pipeline:
    """
    From a set of training examples, fits an ngram TFIDF vectorizer and a multilabel SVM leveraging its output
    """

    vectorizer = TfidfVectorizer(
        strip_accents="ascii",
        analyzer="char",
        ngram_range=(
            hyperparameters.ngram_lower_bound,
            hyperparameters.ngram_upper_bound,
        ),
        max_features=hyperparameters.max_tfidf_features,
    )
    svm = LinearSVC(loss="squared_hinge", C=hyperparameters.C, random_state=RANDOM_SEED)
    clf = MultiOutputClassifier(svm)

    if hyperparameters.scale_features:
        steps = [
            ("tfidf", vectorizer),
            ("scaler", MaxAbsScaler(copy=False)),
            ("classifier", clf),
        ]
    else:
        steps = [("tfidf", vectorizer), ("classifier", clf)]

    final_model = Pipeline(steps=steps)

    training_texts = [
        make_inference_text(ex.instance, hyperparameters.use_abstract)
        for ex in training_examples
    ]
    labels = np.array(
        [labels_to_multihot(ex.labels.foses) for ex in training_examples],
        dtype=np.uint,
    )

    final_model.fit(training_texts, labels)

    return final_model
