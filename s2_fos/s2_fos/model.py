import numpy as np
import os
import sys
import pickle
import fasttext
from typing import List
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from s2_fos.utils import detect_language, make_inference_text


# linearSVC has no predict_proba and MultiOutputClassifier has no decision_function
# so we need this little wrapper
class MultiOutputClassifierWithDecision(MultiOutputClassifier):
    def decision_function(self, X):
        results = [estimator.decision_function(X) for estimator in self.estimators_]
        return np.array(results).squeeze().T  # num_examples X num_classes


class S2FOS:
    """
    Loads in a trained classifier and TFIDF vectorizer.
    Used to produce batches of classification predictions.
    """

    def __init__(self, data_dir):

        setattr(
            sys.modules["__main__"],
            "MultiOutputClassifierWithDecision",
            MultiOutputClassifierWithDecision,
        )

        # to do: update file names when we change the file names
        with open(os.path.join(data_dir, "best_model_use_venue__false.pickle"), "rb") as f:
            self._classifier = pickle.load(f)

        self._fasttext = fasttext.load_model(os.path.join(data_dir, "lid.176.bin"))

        with open(os.path.join(data_dir, "feature_pipe_use_venue__false.pickle"), "rb") as f:
            self._feature_pipe, self._mlb = pickle.load(f)

        # ensure that cached_dict has been built on mlb
        self._mlb._cached_dict = dict(zip(self._mlb.classes_, range(len(self._mlb.classes_))))
        self.mlb_inverse_dict = {v: k for k, v in self._mlb._cached_dict.items()}

    def decision_function(self, papers):
        """Decision scores for a list of papers

        Args:
            papers (list[dict]): A list of dictionaries with 'title' and 'abstract as keys

        Returns:
            scores (np.array): scores for each class for each paper
        """
        texts = [make_inference_text(paper) for paper in papers]

        featurized_text = self._feature_pipe.transform(texts)
        scores = self._classifier.decision_function(featurized_text)

        return self.convert_score_row_to_dict(scores, self.mlb_inverse_dict)

    def predict(self, papers):
        """Predictions scores for a list of papers.
        English detection is done first, and then this rule is applied to each prediction:

        If paper is English:
            If abstract exists:
                If any scores > -0.2:
                    Take all predictions with score > -0.2
                Else:
                    Take first prediction with score > -1.0
            If no abstract exists:
                Take first prediction with score > -0.2
        Else:
            No predictions

        Args:
            papers (list[dict]): A list of dictionaries with 'title' and 'abstract as keys

        Returns:
            scores (np.array): predictions above the thresholds for each class for each paper
        """
        texts = [make_inference_text(paper) for paper in papers]
        english_flag = np.array([detect_language(self._fasttext, text)[1] for text in texts])
        has_abstract = np.array([bool("abstract" in paper and paper["abstract"] is not None) for paper in papers])
        decision_outputs = self.decision_function(papers)
        predictions = []
        for decision_output, english, abstract in zip(decision_outputs, english_flag, has_abstract):
            pred = {}
            if english:
                classes = np.array(list(decision_output.keys()))
                scores = np.array(list(decision_output.values()))
                biggest_score_loc = np.argmax(scores)
                biggest_score = scores[biggest_score_loc]
                if abstract:
                    if biggest_score > -0.2:
                        scores_above = scores > -0.2
                        pred = {classes[i]: scores[i] for i in np.where(scores_above)[0]}
                    elif biggest_score > -1.0:
                        pred[classes[biggest_score_loc]] = biggest_score
                else:
                    if biggest_score > -0.2:
                        pred[classes[biggest_score_loc]] = biggest_score
            predictions.append(pred)
        return predictions

    @staticmethod
    def convert_score_row_to_dict(scores, mlb_inverse_dict):
        """Convert a row of scores to a dictionary of predictions.

        Args:
            scores (np.array): scores for each class for each paper
            mlb_inverse_dict (dict): mapping of class index to class name

        Returns:
            predictions (dict): predictions for each class for each paper
        """
        predictions = []
        for score_row in scores:
            pred = {}
            for i in mlb_inverse_dict.keys():
                pred[mlb_inverse_dict[i]] = score_row[i]
            predictions.append(pred)
        return predictions
