import os
import unittest

import numpy as np

from model.example import Example
from model.instance import Instance
from model.labels import LABELS
from model.prediction import Prediction
from model import utils


FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


class TestUtils(unittest.TestCase):
    def test_make_inference_text__yields_title_only_if_use_abstract_False(self):
        instance = Instance(title="asdf", abstract=None)

        self.assertEqual(utils.make_inference_text(instance, False), "asdf")

    def test_make_inference_text__yield_title_only_if_no_abstract_avail(self):
        instance = Instance(title="asdf", abstract=None)

        self.assertEqual(utils.make_inference_text(instance, True), "asdf")

    def test_make_inference_text__yield_concatenated_text(self):
        instance = Instance(title="asdf", abstract="fdsa")

        self.assertEqual(utils.make_inference_text(instance, True), "asdf fdsa")

    def test_labels_to_multihot(self):
        labels = ["Philosophy", "Geology", "Economics", "Art"]

        multihot = utils.labels_to_multihot(labels)

        self.assertEqual(
            [
                True,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
            ],
            multihot,
        )

    def test_multihot_to_labels(self):
        allhot = np.array([1] * len(LABELS))
        self.assertEqual(LABELS, utils.multihot_to_labels(allhot))

        nonehot = np.array([0] * len(LABELS))
        self.assertEqual([], utils.multihot_to_labels(nonehot))

        halfhot = np.array(
            ([1] * (len(LABELS) // 2)) + ([0] * (len(LABELS) - len(LABELS) // 2))
        )
        self.assertEqual(LABELS[: len(LABELS) // 2], utils.multihot_to_labels(halfhot))

    def test_load_labeled_data(self):
        fixture_data = os.path.join(FIXTURE_DIR, "example_labeled_data")
        results = utils.load_labeled_data(fixture_data)
        expected = [
            Example(
                instance=Instance(title="asdf", abstract=None),
                labels=Prediction(foses=["Art"]),
            ),
            Example(
                instance=Instance(title="fdsa", abstract="some abstract"),
                labels=Prediction(foses=["Biology", "Sociology"]),
            ),
            Example(
                instance=Instance(title="qwer", abstract="some other abstract"),
                labels=Prediction(foses=["Psychology", "Business"]),
            ),
        ]

        self.assertEqual(expected, results)
