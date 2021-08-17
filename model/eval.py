import os
from typing import Any, Dict, List, Optional

from pydantic import BaseSettings, Field

from model.example import Example
from model.prediction import Prediction


class EvalSettings(BaseSettings):
    artifacts_dir: str = Field(
        description="Directory to find model artifacts such as learned model parameters, etc",
        default="/opt/ml/model",
    )
    input_data_dir: str = Field(
        description="Directory to find data files", default="/opt/ml/input/data"
    )
    channel_name: str = Field(
        description="Name of data channel to run evaluation against. Name of subdirectory under `input_data_dir`",
    )
    output_data_dir: str = Field(
        description="Directory to write metrics results into", default="/opt/ml/output"
    )
    model_version: Optional[str] = Field(
        description="Logical name for a model version or experiment, to segment and retrieve artifacts",
        default=None,
    )

    def eval_data_dir(self) -> str:
        return os.path.join(self.input_data_dir, self.channel_name)

    def target_artifacts_dir(self) -> str:
        """Specifies full directory path for location of model artifacts"""

        if self.model_version is None or self.model_version == "":
            return self.artifacts_dir

        return os.path.join(self.artifacts_dir, self.model_version)

    def metrics_output_dir(self) -> str:
        """Specifies full directory path for saving out evaluation metrics"""

        if self.model_version is None or self.model_version == "":
            return os.path.join(self.artifacts_dir, self.channel_name)

        return os.path.join(self.output_data_dir, self.model_version, self.channel_name)


def generate_metrics(
    eval_examples: List[Example], prediction: List[Prediction]
) -> Dict[str, Any]:
    # TODO: what metrics do we need to generate?

    return {"asdf": "fdsa"}
