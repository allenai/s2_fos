"""
Interface class for the S2 FoS model
"""

import numpy as np
import torch

import os
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
)
from typing import List, Optional, Dict

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

from s2_fos.constants import LABELS, MODEL_NAME, TOKENIZER_MODEL_NAME, PROJECT_ROOT_PATH
from s2_language_detection.language_classifier import LanguageClassifier


class Instance(BaseModel):
    """
    Describes one Instance over which the model performs inference.

    The fields below are examples only; please replace them with
    appropriate fields for your model.

    To learn more about declaring pydantic model fields, please see:
    https://pydantic-docs.helpmanual.io/
    """

    text_title: str = Field(description="Title of the paper")
    text_abstract: Optional[str] = Field(description="Abstract of the paper", default=None)
    text_journal_name: Optional[str] = Field(description="Journal name of the paper", default=None)
    text_venue_name: Optional[str] = Field(description="Venue name of the paper", default=None)


class Score(BaseModel):
    """
    Describes the outcome of inference for one Instance

    The fields below are examples only; please replace them with
    appropriate fields for your model.

    To learn more about declaring pydantic model fields, please see:
    https://pydantic-docs.helpmanual.io/
    """

    label: str = Field(description="Predicted fields of study for the paper")
    score: float = Field(description="Confidence scores for each field of study")

    def to_dict(self):
        return {self.label, self.score}


class Prediction(BaseModel):
    """
    Describes the outcome of inference for one Instance

    The fields below are examples only; please replace them with
    appropriate fields for your model.

    To learn more about declaring pydantic model fields, please see:
    https://pydantic-docs.helpmanual.io/
    """

    field_of_studies_predicted_above_threshold: List[str] = Field(
        description="Predicted fields of study for the paper"
    )
    scores: List[Score] = Field(description="Confidence scores for each field of study")


class PredictorConfig(BaseSettings):
    """
    Configuration required by the model to do its work.
    Uninitialized fields will be set via Environment variables.

    The fields below are examples only; please replace them with ones
    appropriate for your model. These serve as a record of the ENV
    vars the consuming application needs to set.
    """

    thr_1: float = Field(default=0.552, description="Threshold for the first level")
    thr_2: float = Field(default=0.621, description="Threshold for the second level")
    thr_3: float = Field(default=0.7, description="Threshold for the second level")
    thr_1_no_abstract: float = Field(default=0.621, description="Threshold for the first level no abstract")
    thr_2_no_abstract: float = Field(default=0.655, description="Threshold for the second level no abstract")
    thr_3_no_abstract: float = Field(default=0.7, description="Threshold for the third level no abstract")


class S2FOS:
    """
    Interface on to your underlying model.
    """

    _config: PredictorConfig
    data_dir: str

    def __init__(self, data_dir: str = 'data'):
        self._config = PredictorConfig()
        self.data_dir = os.path.join(PROJECT_ROOT_PATH, data_dir)
        # Load the language classifier
        self._model_lan_classifier = LanguageClassifier(data_dir=self.data_dir)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Define the model name and check if the model exists locally
        model_path = os.path.join(self.data_dir, 'pytorch_model.bin')
        config_path = os.path.join(self.data_dir, 'config.json')

        # Check if the model and config files exist locally
        if os.path.exists(model_path) and os.path.exists(config_path):
            print("Loading the model from the data_dir")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.data_dir, config=config_path)
        else:
            print("Model not found locally. Downloading from Hugging Face model hub.")
            self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_NAME,
                                                       use_fast=True)

        # Move the model to the appropriate device
        self.model.to(self.device)

    def set_labels(self,
                   threshold_list_np: np.array,
                   abstract_np: np.array,
                   ) -> np.array:
        """
        Threshold values are selected based on the max F1
        Best micro-f1 score: ({'thr_1': 0.5172413793103449, 'thr_2': 0.5517241379310345, 'thr_3': 0.5862068965517241},
        {'micro': 0.7919858573954036, 'macro': 0.7773981933829964, 'weighted': 0.8000682133515643})
        micro: 0.85443
        macro: 0.83920
        weighted: 0.87635

        For abstract missing
        Best micro-f1 score: ({'thr_1': 0.5172413793103449, 'thr_2': 0.6206896551724138, 'thr_3': 0.6206896551724138},
        {'micro': 0.7784503631961258, 'macro': 0.7374930547135901, 'weighted': 0.7847926929179073})
        Average precision:
        micro: 0.84874
        macro: 0.81286
        weighted: 0.86451
        Args:
            threshold_list_np ():
            abstract_np ():

        Returns:

        """
        argmax_idx = np.argpartition(-threshold_list_np, kth=5, axis=1)[:, :5]
        assigned_labels = np.zeros(threshold_list_np.shape)
        for row_idx, row in enumerate(threshold_list_np):
            for idx_n, idx in enumerate(argmax_idx[row_idx]):
                abstract = abstract_np[row_idx]
                if abstract is None or abstract == "":
                    if self._config.thr_1_no_abstract is not None or self._config.thr_2_no_abstract is not None:
                        thr_1, thr_2, thr_3 = (self._config.thr_1_no_abstract, self._config.thr_2_no_abstract,
                                               self._config.thr_3_no_abstract)
                    else:
                        thr_1, thr_2, thr_3 = (thr_1_no_abstract, thr_2_no_abstract, thr_3_no_abstract)
                else:
                    if self._config.thr_1 is not None or self._config.thr_2 is not None:
                        thr_1, thr_2, thr_3 = self._config.thr_1, self._config.thr_2, self._config.thr_3
                    else:
                        thr_1, thr_2, thr_3 = thr_1_w_abstract, thr_2_w_abstract, thr_3_w_abstract

                if row[idx] >= thr_1 and idx_n == 0:
                    assigned_labels[row_idx, idx] = 1
                elif row[idx] >= thr_2 and idx_n >= 1:
                    assigned_labels[row_idx, idx] = 1
                elif row[idx] >= thr_3 and idx_n >= 2:
                    assigned_labels[row_idx, idx] = 1
                else:
                    assigned_labels[row_idx, idx] = 0
        return assigned_labels

    def predict_labels_from_np(self, data_np: np.array) -> np.array:
        """
        Predicts probabilities of the FoS
        Args:
            data_np (np.array): numpy array of the title, abstract, journal_names

        Returns: numpy array of probability scores for each of the fields of study

        """
        text = [f'{self.tokenizer.sep_token}'.join([(example[i] if example[i] is not None else '')
                                                    for i in range(3)]) for example in data_np]
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )

        if "token_type_ids" not in inputs:
            inputs.pop("token_type_ids", None)

        inputs = {key: tensor.to(self.device) for key, tensor in inputs.items()}
        outputs = self.model(**inputs)
        predictions = torch.sigmoid(outputs.logits)
        return np.array(predictions.detach().cpu().numpy())

    def predict_batch(self, instances: List[Instance]) -> List[Prediction]:
        """
        Method called by the client application. One or more Instances will
        be provided, and the caller expects a corresponding Prediction for
        each one.

        If your model gets performance benefits from batching during inference,
        implement that here, explicitly.

        Otherwise, you can leave this method as-is and just implement
        `predict_one()` above. The default implementation here passes
        each Instance into `predict_one()`, one at a time.

        The size of the batches passed into this method is configurable
        via environment variable by the calling application.
        """
        predictions = []
        as_np_array = np.array([
            [
                instance.text_title,
                instance.text_abstract,
                instance.text_journal_name if instance.text_journal_name is not None else instance.text_venue_name,
            ] for instance in instances
        ], dtype=str)
        language_predictions = self._model_lan_classifier.predict(as_np_array[:, :2])
        # Predicting the labels
        raw_predictions = self.predict_labels_from_np(as_np_array)
        labels = self.set_labels(raw_predictions, abstract_np=as_np_array[:, 1])

        # Predicting the fields of study
        for idx, label_row in enumerate(labels.tolist()):
            # Create Score objects for all fields of study with their corresponding scores
            all_fos_scores = [Score(label=LABELS[i], score=float(score)) for i, score in
                              enumerate(raw_predictions[idx].tolist())]

            # Sort all_fos_scores by score in descending order
            all_fos_scores_sorted = sorted(all_fos_scores, key=lambda x: x.score, reverse=True)

            # Check if the paper is in English
            if language_predictions[idx][0] != "en":
                predictions.append(
                    Prediction(field_of_studies_predicted_above_threshold=[], scores=all_fos_scores_sorted)
                )
            else:
                # Filter out the fields of study that are above the threshold and sort them by score
                fos_above_threshold = [score.label for score in all_fos_scores_sorted if
                                       label_row[LABELS.index(score.label)]]

                predictions.append(
                    Prediction(field_of_studies_predicted_above_threshold=fos_above_threshold,
                               scores=all_fos_scores_sorted)
                )
        return predictions

    def convert_dict_to_instances(self, papers: List[Dict[str, str]]) -> List[Instance]:
        return [Instance(text_title=paper.get('title', ''),
                         text_abstract=paper.get('abstract', ''),
                         text_journal_name=paper.get('journal_name', ''),
                         text_venue_name=paper.get('venue_name', '')) for paper in papers]

    def predict(self, papers: List[Dict[str, str]]):
        instances = self.convert_dict_to_instances(papers)
        return {'scores': [[score.to_dict() for score in prediction.scores]
                           for prediction in self.predict_batch(instances)],
                'fields_of_study_predicted': [prediction.field_of_studies_predicted_above_threshold
                                              for prediction in self.predict_batch(instances)]
                }
