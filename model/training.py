from typing import List, Tuple

import numpy as np
from pydantic import BaseModel, BaseSettings, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC

from model.hyperparameters import ModelHyperparameters
from model.instance import Instance
from model.prediction import Prediction
from model.utils import labels_to_multihot, make_inference_text


RANDOM_SEED = 1337  # as is tradition


class Example(BaseModel):
    instance: Instance
    labels: Prediction


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
        [labels_to_multihot(ex.labels.fields_of_study) for ex in training_examples],
        dtype=np.uint,
    )

    final_model.fit(training_texts, labels)

    return final_model
