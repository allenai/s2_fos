import unittest

from model.predictor import PredictorConfig


class TestPredictorConfig(unittest.TestCase):
    def test_model_artifacts_dir(self):
        config = PredictorConfig(artifacts_dir="/some/artifacts", model_version=None)
        self.assertEqual("/some/artifacts", config.model_artifacts_dir())

        config = PredictorConfig(artifacts_dir="/some/artifacts", model_version="")
        self.assertEqual("/some/artifacts", config.model_artifacts_dir())
