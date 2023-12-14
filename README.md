# s2_fos

Code for the model of Semantic Scholar's paper Field of Study classifier.

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
# Download Langauge identification model from [fasttext](https://fasttext.cc/docs/en/language-identification.html)
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin .
# Download model and the model config from huggingface
wget https://huggingface.co/allenai/scibert_scivocab_uncased_fielf_of_study/resolve/main/pytorch_model.bin?download=true .
wget https://huggingface.co/allenai/scibert_scivocab_uncased_fielf_of_study/resolve/main/config.json?download=true .
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

Python file train_net.py contains code for fine tuning the model

## Training

Fine tuning of the model should be done on appropriate GPU instance

```bash
python train_net.py --train_data <path to training data> --test_data <path to test data> \
--val_data <path to validation data> --text_fields title abstract journal_name  --save_path <output_path> --train True \
--model_checkpoint_path <model_check_point_path>  --project_name <weights and biases project name>
--batch_size <batch size> --learning_rate <learning rate> --warmup_ratio <warm up ratio> \
--wandb_name <weights and biases run name> --wandb_run_des <run description> --log_dir <log directory>
```
