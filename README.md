# s2_fos

Code for scoring/training/training data generation of Semantic Scholar's paper Field of Study classifier.

Model uses fined-tuned SciBERT model to predict the field of study for a given paper.

During inference
-- First language of the paper is determined. If it is an English paper FoS fields and scores are predicted

## Hugging face artifacts
Model weights and config can be found under: [allenai/scibert_scivocab_uncased_field_of_study](https://huggingface.co/allenai/scibert_scivocab_uncased_field_of_study)
Training data/annotations/openAI response can be found under: [allenai/fos_model_training_data_open_ai_annotations](https://huggingface.co/datasets/allenai/fos_model_training_data_open_ai_annotations)

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
If you have problems installing poetry refer to the [poetry documentation](https://python-poetry.org/docs/#installation)

To obtain the necessary data, run these commands after the package is installed:

```bash
# Download Langauge identification model from [fasttext](https://fasttext.cc/docs/en/language-identification.html)
cd data & wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
```
Location of the model on hugging face is [allenai/scibert_scivocab_uncased_fielf_of_study](https://huggingface.co/allenai/scibert_scivocab_uncased_fielf_of_study)
Before running the code, make sure to accept the license agreement.

Once license agreement is accepted, you need to generate hugging face token and set it as HUGGINGFACE_HUB_TOKEN
environment variable.
```bash
export HUGGINGFACE_HUB_TOKEN=<your token>
```

## Inference Example Code

```python
from s2_fos import S2FOS

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

predictor = S2FOS()
print(predictor.predict(papers))
```
## Development

To run the tests, run

```bash
poetry shell
poetry run pytest
```
Code in the test directory is almost identical to the above example code.

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
Fine-tuned model weights are available at [allenai/scibert_scivocab_uncased_field_of_study](https://huggingface.co/allenai/scibert_scivocab_uncased_field_of_study)

## How to fine-tune the model

To fine tune the model 
- Download the training data from [allenai/fos_model_training_data_open_ai_annotations](https://huggingface.co/datasets/allenai/fos_model_training_data_open_ai_annotations) under fos_open_ai_labels.parquet
- Split the data into train, test and validation sets save them into separate parquet files
- Update run.sh script by replacing the paths to the training, test and validation data with the paths to the files you created in the previous step
- Update any other parameters in the run.sh script as needed replace <> with the appropriate values
- Run the training code with the following parameters:
```bash
cd src/s2_fos/training
poetry shell
bash run.sh
```