"""
Implement your model here! Or not.
"""
import os

import torch
import numpy as np
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
)


class PredictProbabilities:

    def __init__(self, model_path):
        # Load the trained model and tokenizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if 'allenai/scibert_scivocab_uncased' not in model_path:
            full_model_path = os.path.join(model_path, 'pytorch_model.bin')

        self.config = AutoConfig.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            full_model_path, config=self.config)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased',
                                                       use_fast=True)

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
