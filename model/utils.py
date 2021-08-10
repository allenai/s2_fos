import os
import pickle
from typing import List, Tuple

import numpy as np
from sklearn.pipeline import Pipeline

from model.hyperparameters import ModelHyperparameters
from model.labels import LABELS
from model.instance import Instance


HYPERPARAMETERS_FNAME = "hyperparameters.json"
CLASSIFIER_FNAME = "classifier.pkl"


def make_inference_text(instance: Instance, use_abstract: bool) -> str:
    """Makes the combined text to perform inference over, from an Instance"""
    if use_abstract and instance.abstract:
        return f"{instance.title} {instance.abstract}"

    return instance.title


def labels_to_multihot(fields_of_study: List[str]) -> List[bool]:
    """Generates a multi-hot vector for Fields of Studies"""
    label_set = set(fields_of_study)
    return [label in label_set for label in LABELS]


def multihot_to_labels(multihot: np.ndarray) -> List[str]:
    """Converts model output vector to str label list"""
    labels = []

    for index, label in enumerate(LABELS):
        if multihot[index] == 1:
            labels.append(label)

    return labels


def save_model(
    artifacts_dir: str, hyperparameters: ModelHyperparameters, classifier: Pipeline
) -> None:
    """
    Saves out model hyperparameters and trained classifier to a target directory.
    """
    with open(os.path.join(artifacts_dir, HYPERPARAMETERS_FNAME), "w") as fhyper:
        fhyper.write(hyperparameters.json())

    with open(os.path.join(artifacts_dir, CLASSIFIER_FNAME), "wb") as fclassifier:
        pickle.dump(classifier, fclassifier)


def load_model(artifacts_dir) -> Tuple[ModelHyperparameters, Pipeline]:
    """
    Loads in previously saved hyperparameters and trained classifier from a target directory.
    """
    hyperparameters = ModelHyperparameters.parse_file(
        os.path.join(artifacts_dir, HYPERPARAMETERS_FNAME)
    )
    classifier = pickle.load(open(os.path.join(artifacts_dir, CLASSIFIER_FNAME), "rb"))

    return hyperparameters, classifier
