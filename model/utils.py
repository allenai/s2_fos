import logging
import os
import pickle
from typing import List, Optional, Tuple

import numpy as np
from sklearn.pipeline import Pipeline

from model.example import Example
from model.hyperparameters import ModelHyperparameters
from model.instance import Instance
from model.labels import LABELS


logger = logging.getLogger(__name__)

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


def load_labeled_data(labeled_data_dir: str) -> List[Example]:
    """
    Given a directory of labeled data JSONL files, loads
    a list of examples.
    """

    examples: List[Example] = []

    training_files = [
        f
        for f in os.listdir(labeled_data_dir)
        if os.path.isfile(os.path.join(labeled_data_dir, f)) and f.endswith(".jsonl")
    ]

    for training_file in training_files:
        logger.info("Loading training data from `{f}`")
        with open(os.path.join(labeled_data_dir, training_file), "r") as f:
            for line in f:
                example = Example.parse_raw(line)
                examples.append(example)

    return examples


def save_model(
    target_dir: str, hyperparameters: ModelHyperparameters, classifier: Pipeline
) -> None:
    """
    Saves out model hyperparameters and trained classifier to a target directory.
    """

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    with open(os.path.join(target_dir, HYPERPARAMETERS_FNAME), "w") as fhyper:
        fhyper.write(hyperparameters.json(indent=4))

    with open(os.path.join(target_dir, CLASSIFIER_FNAME), "wb") as fclassifier:
        pickle.dump(classifier, fclassifier)


def load_model(artifacts_dir: str) -> Tuple[ModelHyperparameters, Pipeline]:
    """
    Loads in previously saved hyperparameters and trained classifier from a target directory.
    """

    logging.info("Loading hyperparameters from disk...")

    hyperparameters = ModelHyperparameters.parse_file(
        os.path.join(artifacts_dir, HYPERPARAMETERS_FNAME)
    )

    logging.info("Loading pickled classifier from disk...")
    classifier = pickle.load(open(os.path.join(artifacts_dir, CLASSIFIER_FNAME), "rb"))

    return hyperparameters, classifier
