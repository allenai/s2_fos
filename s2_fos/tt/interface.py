"""
This file contains the classes required by Semantic Scholar's
TIMO tooling.

You must provide a wrapper around your model, as well
as a definition of the objects it expects, and those it returns.
"""

from typing import List, Optional
from pydantic import BaseModel, BaseSettings, Field
from s2_fos import S2FOS, make_inference_text, detect_language


class Instance(BaseModel):
    """Represents one paper for which we can predict fields of study"""

    title: str = Field(description="Title text for paper")
    abstract: Optional[str] = Field(description="Abstract text for paper (optional)")


class DecisionScore(BaseModel):
    """Represents decision score predicted for a given field of study for a given paper"""

    label: str
    score: float


class Prediction(BaseModel):
    """Represents predicted scores for all fields of study for a given paper"""

    scores: List[DecisionScore] = Field(description="Decision scores for all fields of study")


class PredictorConfig(BaseSettings):
    """
    The set of configuration parameters required to instantiate a predictor and
    initialize it for inference. This is an appropriate place to specify any parameter
    or configuration values your model requires that aren't packaged with your
    versioned model artifacts. These should be rare beyond the included
    `artifacts_dir`.

    Values for these config fields can be provided as environment variables, see:
    `./docker.env`
    """

    model_version: Optional[str] = Field(
        description="Logical name for a model version or experiment, to segment and retrieve artifacts",
        default=None,
    )


class Predictor:
    """
    Interface on to your underlying model.

    This class is instantiated at application startup as a singleton.
    You should initialize your model inside of it, and implement
    prediction methods.

    If you specified an artifacts.tar.gz for your model, it will
    have been extracted to `artifacts_dir`, provided as a constructor
    arg below.
    """

    _config: PredictorConfig
    _artifacts_dir: str

    def __init__(self, config: PredictorConfig, artifacts_dir: str):
        self._config = config
        self._artifacts_dir = artifacts_dir
        self._load_model()

    def _load_model(self) -> None:
        """
        Perform whatever start-up operations are required to get your
        model ready for inference. This operation is performed only once
        during the application life-cycle.
        """
        self.model = S2FOS(self._artifacts_dir)

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
        # expecting a list of dicts instaed of a list of Instances
        papers = [dict(instance) for instance in instances]
        texts = [make_inference_text(paper) for paper in papers]
        english_flag = [detect_language(self.model._fasttext, text)[1] for text in texts]
        decision_scores = self.model.decision_function(papers)
        # now the output should be a prediction aka a list of DecisionScores
        output = []
        for decision_score, english in zip(decision_scores, english_flag):
            if english:
                output.append(
                    Prediction(
                        scores=[DecisionScore(label=label, score=score) for label, score in decision_score.items()]
                    )
                )
            else:
                output.append(Prediction(scores=[]))
        return output


if __name__ == "__main__":

    instances = [
        Instance(
            title="Neural Networks are Great",
            abstract="Neural networks are known to be really great models. You should use them.",
        ),
        Instance(
            title="Cryptozoology for protein-folding metabolomics",
            abstract="We show that cryptozoology is a great way to study protein folding. With 300 patients, we sequence their genomes.",
        ),
        Instance(
            title="The Fate of All Oceans is Decided by the Whales",
        ),
        Instance(
            title="すべてのネットワークの運命は、ランダムシードによって決定されます",
            abstract="ネットワークは、ランダムシードによって決定されます。",
        ),
        Instance(
            title="Precursor charge state prediction for electron transfer dissociation tandem mass spectra.",
            abstract="Electron-transfer dissociation (ETD) induces fragmentation along the peptide backbone by transferring an electron from a radical anion to a protonated peptide. In contrast with collision-induced dissociation, side chains and modifications such as phosphorylation are left intact through the ETD process.",
        ),
        Instance(
            title="Hannnah Arendt's 'Human Condition' or How to Survive in a Men's World",
            abstract="In this paper I want to analyze Hannah Arendt’s concepts, described in her Human Condition, from a perspective which takes into consideration her own fragile identity, placed in a particular way under the sign of the major influence of Martin Heidegger and, generally, under the influence of the men-politicians and men-philosophers. The triangle labor-work-action dissimulates an informal tendency to hide the woman’s condition under the human condition. The feminine and maternal spirit finds its expression here too, protesting against the child and childhood politicizing idea.",
        ),
    ]

    s2fos = Predictor(config=PredictorConfig(), artifacts_dir="./data")

    predictions = s2fos.predict_batch(instances)

    # matching a few things from the s2_fos README
    # first one is CS
    cs_pred_0 = [i for i in predictions[0].scores if i.label == "Computer science"][0]
    assert cs_pred_0 == DecisionScore(label="Computer science", score=-0.21976448315303118)

    # last one is philosophy
    cs_pred_5 = [i for i in predictions[5].scores if i.label == "Philosophy"][0]
    assert cs_pred_5 == DecisionScore(label="Philosophy", score=-0.4015553471152487)
