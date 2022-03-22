"""
Write integration tests for your model interface code here.

The TestCase class below is supplied a `container`
to each test method. This `container` object is a proxy to the
Dockerized application running your model. It exposes a single method:

```
predict_batch(instances: List[Instance]) -> List[Prediction]
```

To test your code, create `Instance`s and make normal `TestCase`
assertions against the returned `Prediction`s.

e.g.

```
def test_prediction(self, container):
    instances = [Instance(), Instance()]
    predictions = container.predict_batch(instances)

    self.assertEqual(len(instances), len(predictions)

    self.assertEqual(predictions[0].field1, "asdf")
    self.assertGreatEqual(predictions[1].field2, 2.0)
```
"""


import logging
import sys
import unittest

from .interface import Instance, DecisionScore


try:
    from timo_interface import with_timo_container
except ImportError as e:
    logging.warning(
        """
    This test can only be run by a TIMO test runner. No tests will run. 
    You may need to add this file to your project's pytest exclusions.
    """
    )
    sys.exit(0)


@with_timo_container
class TestInterfaceIntegration(unittest.TestCase):
    def test__predictions(self, container):
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

        predictions = container.predict_batch(instances)

        # matching a few things from the s2_fos README
        # first one is CS
        cs_pred_0 = [i for i in predictions[0].scores if i.label == "Computer science"][0]
        self.assertEqual(cs_pred_0, DecisionScore(label="Computer science", score=-0.21976448315303118))

        # last one is philosophy
        cs_pred_5 = [i for i in predictions[5].scores if i.label == "Philosophy"][0]
        self.assertEqual(cs_pred_5, DecisionScore(label="Philosophy", score=-0.4015553471152487))
