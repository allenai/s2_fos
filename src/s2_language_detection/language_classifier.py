import os
import logging
from typing import Union

import fasttext
import numpy.typing as npt

from s2_fos.constants import FASTTEXT_FNAME, PROJECT_ROOT_PATH


class LanguageClassifier:
    def __init__(self, data_dir: str = None):
        self.logger = logging.getLogger(__name__)
        if data_dir is None:
            self.model_path = os.path.join(PROJECT_ROOT_PATH, data_dir, FASTTEXT_FNAME)
        else:
            self.model_path = os.path.join(data_dir, FASTTEXT_FNAME)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Language classifier model {FASTTEXT_FNAME} not found in {PROJECT_ROOT_PATH} \n"
                f"Please download the model from https://fasttext.cc/docs/en/language-identification.html"
                f"and place it in {data_dir}")

        self.fasttext_model = fasttext.load_model(self.model_path)

    def predict(self, input_array: npt.ArrayLike) -> npt.ArrayLike:
        """
        Predictions for an array of inputs
        :param input_array: of the shape 2 X N
        :return: Array of predictions
        """
        result = []
        for text_title_abstract in input_array:
            result.append(self.detect_language(*text_title_abstract))
        return result

    def detect_language(self, text_title: str = None, text_abstract: str = None) -> Union[str, float]:
        """
        Detect the language used in the input text with trained language classifier. If abstract is present, and it does
        not contain 'no abstract' returns language of the abstract and the soft max_value prediction. Returns
        predicted language for the title and corresponding softmax_value
        :param text_title Title of the paper
        :param text_abstract Abstract of the paper
        """
        if text_abstract and 'no abstract' not in text_abstract:
            predicted_language_abstract, soft_max_pred_abstract = self.detect_language_single_entity(text_abstract)

            if predicted_language_abstract != 'un':
                return predicted_language_abstract, soft_max_pred_abstract

        return self.detect_language_single_entity(text_title)

    def detect_language_single_entity(self, text: str) -> Union[str, float]:
        """
        Detect the language used in the input text with trained language classifier.
        Returns 'un' if the text provided has less than 2 words, except ['zh', 'jp']
        :param text to predict language for
        """
        if not text:
            return 'un', 0.0

        fasttext_pred = self.fasttext_model.predict(text.lower().replace('\n', ' '))
        predicted_language_ft = fasttext_pred[0][0].split('__')[-1]
        soft_max_pred = fasttext_pred[1][0]

        # Chinese and japanese do not use white space for word separation, thus we first check if the predicted language
        # is not one from the list before checking how many words the text contains.
        if predicted_language_ft not in ['zh', 'jp'] and len(text.split()) <= 1:
            return 'un', 0.0

        return predicted_language_ft, soft_max_pred
