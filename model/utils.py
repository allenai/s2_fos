import logging
import os
import pickle
import re
import sys
from text_unidecode import unidecode
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from model.example import Example
from model.hyperparameters import ModelHyperparameters
from model.instance import Instance
from model.labels import LABELS
from model.multioutput import MultiOutputClassifierWithDecision

import fasttext
import pycld2 as cld2

logger = logging.getLogger(__name__)

HYPERPARAMETERS_FNAME = "hyperparameters.json"
FEATURIZER_FNAME = "feature_pipe_use_venue__false.pickle"
CLASSIFIER_FNAME = "best_model_use_venue__false.pickle"
FASTTEXT_FNAME = "lid.176.bin"

ACCEPTABLE_CHARS = re.compile(r"[^a-zA-Z\s]+")

FASTTEXT_MODEL = fasttext.load_model(
    os.path.join(str(os.environ.get("ARTIFACTS_DIR")), FASTTEXT_FNAME)
)


def make_inference_text(instance: Instance, use_abstract: bool) -> str:
    """Makes the combined text to perform inference over, from an Instance"""
    if use_abstract and instance.abstract:
        return concat_text(instance)

    return normalize_text(instance.title)


def normalize_text(text):
    """
    Normalize text.
    Parameters
    ----------
    text: string
        the text to normalize
    special_case_apostrophie: bool
        whether to replace apostrophes with empty strings rather than spaces
    Returns
    -------
    string: the normalized text
    """
    if text is None or len(text) == 0:
        return ""

    norm_text = unidecode(text).lower()
    norm_text = ACCEPTABLE_CHARS.sub(" ", norm_text)
    norm_text = re.sub(r"\s+", " ", norm_text).strip()

    return norm_text


def concat_text(instance: Instance, sep="|", sep_num=5):

    title = normalize_text(instance.title)
    abstract = normalize_text(instance.abstract)

    return f"{title} {sep * sep_num} {abstract}"


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


def load_model(
    artifacts_dir: str,
) -> Tuple[ModelHyperparameters, MultiOutputClassifierWithDecision]:
    """
    Loads in previously saved hyperparameters and trained classifier from a target directory.
    """

    logging.info("Loading hyperparameters from disk...")

    hyperparameters = ModelHyperparameters.parse_file(
        os.path.join(artifacts_dir, HYPERPARAMETERS_FNAME)
    )

    setattr(
        sys.modules["__main__"],
        "MultiOutputClassifierWithDecision",
        MultiOutputClassifierWithDecision,
    )

    logging.info("Loading pickled classifier from disk...")
    classifier = pickle.load(open(os.path.join(artifacts_dir, CLASSIFIER_FNAME), "rb"))

    return hyperparameters, classifier


def load_feature_pipe(artifacts_dir: str) -> Tuple[Pipeline, MultiLabelBinarizer]:
    feature_pipe, mlb = pickle.load(
        open(os.path.join(artifacts_dir, FEATURIZER_FNAME), "rb")
    )

    return feature_pipe, mlb


def detect_language(text: str):
    """
    Detect the language used in the input text with trained language classifer.
    """
    if len(text.split()) <= 1:
        return (False, False, "un")

    # fasttext
    isuppers = [c.isupper() for c in text if c.isalpha()]
    if len(isuppers) == 0:
        return (False, False, "un")
    elif sum(isuppers) / len(isuppers) > 0.9:
        fasttext_pred = FASTTEXT_MODEL.predict(text.lower().replace("\n", " "))
        predicted_language_ft = fasttext_pred[0][0].split("__")[-1]
    else:
        fasttext_pred = FASTTEXT_MODEL.predict(text.replace("\n", " "))
        predicted_language_ft = fasttext_pred[0][0].split("__")[-1]

    # cld2
    try:
        cld2_pred = cld2.detect(text)
        predicted_language_2 = cld2_pred[2][0][1]
        if predicted_language_2 == "un":
            predicted_language_2 = "un_2"
    except:  # noqa: E722
        predicted_language_2 = "un_2"

    if predicted_language_ft == "un_ft" and predicted_language_2 == "un_2":
        predicted_language = "un"
        is_reliable = False
    elif predicted_language_ft == "un_ft":
        predicted_language = predicted_language_2
        is_reliable = True
    elif predicted_language_2 == "un_2":
        predicted_language = predicted_language_ft
        is_reliable = True
    elif predicted_language_2 != predicted_language_ft:
        predicted_language = "un"
        is_reliable = False
    else:
        predicted_language = predicted_language_2
        is_reliable = True

    # is_english can now be obtained
    is_english = predicted_language == "en"

    return is_reliable, is_english, predicted_language
