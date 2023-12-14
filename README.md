# s2_fos

Model code for Semantic Scholar's paper Field of Study classifier.

Model uses fined-tuned SciBERT model to predict the field of study for a given paper.

## Installation
To install this package, run the following:

```bash
git clone https://github.com/allenai/s2_fos.git
cd s2_fos
conda create -y --name s2_fos python==3.8
conda activate s2_fos
# Install poetry
curl -sSL https://install.python-poetry.org | python3 -
poetry install
```

To obtain the necessary data, run these commands after the package is installed:

```bash
cd data
# Download Langauge indentification model from [fasttext](https://fasttext.cc/docs/en/language-identification.html)
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin .
cd ..
```


## Example

```python
from s2_fos.tt.interface import Instance, PredictorConfig, Predictor

model_path = './data'

instance = Instance(text_title="A Prototype-Based Few-Shot Named Entity Recognition",
                    abstract="Few-shot Named Entity Recognition (NER) task focuses on identifying name entities on a "
                             "small amount of supervised training data. The work based on prototype network shows "
                             "strong adaptability on the few-shot NER task. We think that the core idea of these "
                             "approaches is to learn how to aggregate the representation of token mappings in vector "
                             "space around entity class. But, as far as we know, no such work has been investigated its"
                             " effect. So, we propose the ClusLoss and the ProEuroLoss aiming to enhance the model's "
                             "ability in terms of aggregating semantic information spatially, thus helping the model "
                             "better distinguish entity types. Experimental results show that ProEuroLoss achieves "
                             "state-of-the-art performance on the average F1 scores for both 1-shot and 5-shot NER "
                             "tasks, while the ClusLoss has competitive performance on such tasks.', "
                             "journal_name='Proceedings of the 8th International Conference on Computing and Artificial "
                             "Intelligence")
config = PredictorConfig()

predictor = Predictor(config, artifacts_dir=model_path)

predictions = predictor.predict_batch([instance])
print(predictions)
```
