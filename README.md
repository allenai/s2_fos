# s2_fos

Model code for Semantic Scholar's paper Field of Study classifier.

Uses a multi-label SVM classifier over paper title and abstract text,
embedded as character-level ngrams.

## Installation
To install this package, run the following:

```bash
git clone https://github.com/allenai/s2_fos.git
cd s2_fos
conda create -y --name s2_fos python==3.8
conda activate s2_fos
python setup.py develop
```

To obtain the necessary data, run these command afters the package is installed:

```bash
cd data
aws s3 cp --no-sign-request s3://ai2-s2-research/s2_fos_artifacts_v001.tar.gz .
tar -xvzf s2_fos_artifacts_v001.tar.gz
```


## Example
An example of how to use this repo:

```python
from s2_fos import S2FOS, LABELS

# what is the space of labels?
print(LABELS)

# the data is a list of dictionaries
papers = [
    {
        'title': 'Neural Networks are Great',
        'abstract': 'Neural networks are known to be really great models. You should use them.',
    },
    {
        'title': 'Cryptozoology for protein-folding metabolomics',
        'abstract': 'We show that cryptozoology is a great way to study protein folding. With 300 patients, we sequence their genomes.',
    },
    {
        'title': 'The Fate of All Oceans is Decided by the Whales',
    },
    {
        'title': 'すべてのネットワークの運命は、ランダムシードによって決定されます',
        'abstract': 'ネットワークは、ランダムシードによって決定されます。',
    },
    {
        'title': 'Precursor charge state prediction for electron transfer dissociation tandem mass spectra.',
        'abstract': 'Electron-transfer dissociation (ETD) induces fragmentation along the peptide backbone by transferring an electron from a radical anion to a protonated peptide. In contrast with collision-induced dissociation, side chains and modifications such as phosphorylation are left intact through the ETD process.'
    },
    {
        'title': "Hannnah Arendt's 'Human Condition' or How to Survive in a Men's World",
        'abstract': "In this paper I want to analyze Hannah Arendt’s concepts, described in her Human Condition, from a perspective which takes into consideration her own fragile identity, placed in a particular way under the sign of the major influence of Martin Heidegger and, generally, under the influence of the men-politicians and men-philosophers. The triangle labor-work-action dissimulates an informal tendency to hide the woman’s condition under the human condition. The feminine and maternal spirit finds its expression here too, protesting against the child and childhood politicizing idea."
    }
]

# load the model, point to the artifacts downloaded from s3
data_dir = 'data/'
s2ranker = S2FOS(data_dir)


# get the predictions, which includes the following rules
# If paper is English:
#     If abstract exists:
#         If any scores > -0.2:
#             Take all predictions with score > -0.2
#         Else:
#             Take first prediction with score > -1.0
#     If no abstract exists:
#         Take first prediction with score > -0.2
# Else:
#     No predictions
s2ranker.predict(papers)

# also can get the raw decision scores which doesn't apply any of the rules above
s2ranker.decision_function(papers)
```