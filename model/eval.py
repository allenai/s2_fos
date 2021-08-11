import os
from typing import Any, List, Optional

from pydantic import BaseSettings, Field

from model.example import Example
from model.prediction import Prediction


class EvalSettings(BaseSettings):
    artifacts_dir: str = Field(
        description="Directory to find model artifacts such as learned model parameters, etc",
        default="/opt/ml/model",
    )
    input_data_dir: str = Field(
        description="Directory to find data files",
        default="/opt/ml/input/_input_data"
    )
    channel_name: str = Field(
        description="Name of data channel to run evaluation against. Name of subdirectory under `input_data_dir`",
    )
    model_version: Optional[str] = Field(
        description="Logical name for a model version or experiment, to segment and retrieve artifacts",
        default=None
    )

    def eval_data_dir(self) -> str:
        return os.path.join(self.input_data_dir, self.channel_name)

    def target_artifacts_dir(self) -> str:
        """Specifies full directory path for location of model artifacts"""

        if self.model_version is None:
            return self.artifacts_dir

        return os.path.join(self.artifacts_dir, self.model_version)


def generate_metrics(eval_examples: List[Example], prediction: List[Prediction]) -> Any:
    # TODO: what metrics do we need to generate?
    pass


