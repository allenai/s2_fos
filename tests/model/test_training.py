import os
import unittest

from model.training import TrainingConfig
from model.hyperparameters import ModelHyperparameters


FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


class TestTrainingConfig(unittest.TestCase):
    def test_training_data_dir(self):
        config = TrainingConfig(input_data_dir="/asdf/fdsa", channel_name="foobar")
        self.assertEqual("/asdf/fdsa/foobar", config.training_data_dir())

    def test_target_artifacts_dir(self):
        config = TrainingConfig(artifacts_dir="/some/artifacts", model_version=None)
        self.assertEqual("/some/artifacts", config.target_artifacts_dir())

        config = TrainingConfig(artifacts_dir="/some/artifacts", model_version="")
        self.assertEqual("/some/artifacts", config.target_artifacts_dir())

    def test_load_hyperparameters(self):
        config = TrainingConfig(
            input_config_dir=FIXTURE_DIR,
            hyperparameters_file="example_hyperparameters.json",
        )
        hyperparameters = config.load_hyperparameters()
        expected = ModelHyperparameters(
            ngram_lower_bound=1,
            ngram_upper_bound=3,
            max_tfidf_features=10000,
            scale_features=False,
            use_abstract=True,
            C=1.2,
        )

        self.assertEqual(expected, hyperparameters)
