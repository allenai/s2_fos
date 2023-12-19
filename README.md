# s2_fos

Code for scoring/training/training data generation of Semantic Scholar's paper Field of Study classifier.

Model uses fined-tuned SciBERT model to predict the field of study for a given paper.

During inference
-- First language of the paper is determined. If it is an English paper FoS fields and scores are predicted

## Installation
To install this package, run the following:

```bash
git clone https://github.com/allenai/s2_fos.git
cd s2_fos
# Install poetry
curl -sSL https://install.python-poetry.org | python3 -
poetry install
# Activate the virtual environment
poetry shell
## Due to the (non compliance)[https://github.com/python-poetry/poetry/issues/6113] of fasttext with PEPE-518, 
# we need to install it manually
pip install fasttext
```

To obtain the necessary data, run these commands after the package is installed:

```bash
mkdir data
# Download Langauge identification model from [fasttext](https://fasttext.cc/docs/en/language-identification.html)
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin ./data/
# Download model and the model config from huggingface
wget https://huggingface.co/allenai/scibert_scivocab_uncased_fielf_of_study/resolve/main/pytorch_model.bin?download=true ./data/
wget https://huggingface.co/allenai/scibert_scivocab_uncased_fielf_of_study/resolve/main/config.json?download=true ./data/
```


## Inference Example Code

```python
from s2_fos import S2FOS

model_path = './data'

papers = [{'title': "A Prototype-Based Few-Shot Named Entity Recognition",
           'abstract': "Few-shot Named Entity Recognition (NER) task focuses on identifying name entities on a "
                            "small amount of supervised training data. The work based on prototype network shows "
                            "strong adaptability on the few-shot NER task. We think that the core idea of these "
                            "approaches is to learn how to aggregate the representation of token mappings in vector "
                            "space around entity class. But, as far as we know, no such work has been investigated its"
                            " effect. So, we propose the ClusLoss and the ProEuroLoss aiming to enhance the model's "
                            "ability in terms of aggregating semantic information spatially, thus helping the model "
                            "better distinguish entity types. Experimental results show that ProEuroLoss achieves "
                            "state-of-the-art performance on the average F1 scores for both 1-shot and 5-shot NER "
                            "tasks, while the ClusLoss has competitive performance on such tasks.",
            'journal_name': "Proceedings of the 8th International Conference on Computing and Artificial Intelligence",
           'venue_name': ''
           }]

predictor = S2FOS(data_dir=model_path)
print(predictor.predict(papers))
```
## Development

To run the tests, run

```bash
poetry shell
poetry run pytest
```

## Training
Python file train_net.py contains code for model fine-tuning.

To run the training, run the following command on appropriate GPU machine:
```bash
python train_net.py --train_data <path to training data> --test_data <path to test data> \
--val_data <path to validation data> --text_fields title abstract journal_name  --save_path <output_path> --train True \
--model_checkpoint_path <model_check_point_path>  --project_name <weights and biases project name>
--batch_size <batch size> --learning_rate <learning rate> --warmup_ratio <warm up ratio> \
--wandb_name <weights and biases run name> --wandb_run_des <run description> --log_dir <log directory>
```

## Calling OpenAI API to generate training data

To call OpenAI API, you need to set OPENAI_API_KEY environment variable to your API key

Example file is located in src/s2_fos/training/open_ai_prompts.py:
Run it with
```bash
poetry shell
poetry run python ./src/s2_fos/training/open_ai_prompts.py 
```
It reads the data from data/paper_title_abstract_example.json and writes the results to 
data/paper_title_abstract_example_openai.json

The OpenAI prompt is defined in src/s2_fos/training/open_ai_prompts.py

## Training data
Synthetic training data is generated using the code in src/s2_fos/training/open_ai_prompts.py

Can be found in [allenai/fos_model_training_data_open_ai_annotations](https://huggingface.co/datasets/allenai/fos_model_training_data_open_ai_annotations) under fos_open_ai_labels.parquet

## Fine-tuned model weights
Fine-tuned model weights are available at [allenai/scibert_scivocab_uncased_fielf_of_study](https://huggingface.co/allenai/scibert_scivocab_uncased_fielf_of_study)

