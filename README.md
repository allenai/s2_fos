# s2_fos
Code for scoring, training, and data generation for Semantic Scholar's Field of Study (FoS) classifier model.

The model utilizes a fine-tuned SciBERT model to predict the field of study for a given paper.

During inference:
- The language of the paper is first determined. If the paper is in English, the Field of Study categories and their corresponding scores are predicted.

## Installation
To install this package with `poetry`, run the following commands:

```bash
git clone https://github.com/allenai/s2_fos.git
cd s2_fos
# Install poetry
curl -sSL https://install.python-poetry.org | python3 -
poetry install
# Activate the virtual environment
poetry shell
# Due to the non-compliance of fasttext with PEP 518, 
# we need to install it manually
pip install fasttext
```

If you encounter problems installing Poetry, please refer to the [Poetry documentation](https://python-poetry.org/docs/#installation).

Alternatively, you can install with anaconda:
```bash
git clone https://github.com/allenai/s2_fos.git
cd s2_fos
conda create -y --name s2fos python==3.8
conda activate s2fos
pip install -e .
pip install fasttext
```

To obtain the necessary data, run these commands after the package is installed:

```bash
# Download the Language identification model from fasttext
cd data && wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
```

## Hugging Face Artifacts
Model weights, training data, and annotations are available on Hugging Face under the [ImpACT License Low Risk](https://allenai.org/licenses/impact-lr).

Model weights and config can be found under: [allenai/scibert_scivocab_uncased_field_of_study](https://huggingface.co/allenai/scibert_scivocab_uncased_field_of_study)

Training data, annotations, and OpenAI responses can be found under: [allenai/fos_model_training_data_open_ai_annotations](https://huggingface.co/datasets/allenai/fos_model_training_data_open_ai_annotations)

To be able to access the model weights and training data:
- If you don't have one, create an [Hugging Face account](https://huggingface.co).
- If you don't have one, generate [a Hugging Face token](https://huggingface.co/settings/tokens).
- While logged in, click on both of the model weights and data links above and accept the license agreements.

Set your Hugging Face token as an environment variable with the following command (replace `<your_token>` with your actual token):

```bash
export HUGGINGFACE_HUB_TOKEN=<your_token>
```

Or set the token in Python before importing from `s2_fos`:
```
import os
os.environ['HUGGINGFACE_HUB_TOKEN'] = '<your_token>'
```

## Inference Example Code

```python
from s2_fos import S2FOS

# Example paper data
# note that journal_name is just a convention - it can be any venue like NeurIPS, arxiv, etc
papers = [{
    'title': "A Prototype-Based Few-Shot Named Entity Recognition",
    'abstract': ("Few-shot Named Entity Recognition (NER) task focuses on identifying named entities with "
                 "a small amount of supervised training data. Work based on prototype networks shows "
                 "strong adaptability for the few-shot NER task. We believe that the core idea of these "
                 "approaches is to learn how to aggregate the representation of token mappings in vector "
                 "space around entity classes. However, to our knowledge, no work has investigated its "
                 "effect. Therefore, we propose ClusLoss and ProEuroLoss, aiming to enhance the model's "
                 "ability to aggregate semantic information spatially, thus helping the model "
                 "better distinguish between entity types. Experimental results show that ProEuroLoss achieves "
                 "state-of-the-art performance on average F1 scores for both 1-shot and 5-shot NER "
                 "tasks, while ClusLoss has competitive performance in such tasks."),
    'journal_name': "Proceedings of the 8th International Conference on Computing and Artificial Intelligence",
}]

# Initialize the predictor
predictor = S2FOS()

# Predict the fields of study
print(predictor.predict(papers))
```

## Development

To run the tests with anaconda, execute the following commands:

```bash
poetry shell
poetry run pytest
```

Or with anaconda:
```bash
pip install pytest
pytest test
```

## Training

The Python file `train_net.py` contains code for model fine-tuning.

To run the training, execute the following command on an appropriate GPU machine (note that it can also run on a CPU, but it will be very slow):
first you need to replace <parameters> in the src/s2_fos/training/run.sh script with the appropriate values

Next run the training code with the following parameters:
```bash
cd src/s2_fos/training
poetry shell
bash run.sh
```
Training data is downloaded from Hugging Face under the [ImpACT License Low Risk](https://allenai.org/licenses/impact-lr).
Training data is downloaded into `~/.cache/huggingface/hub/datasets--allenai--fos_model_training_data_open_ai_annotations` folder.
The training data is split into train, test, and validation sets with the following ratios: 0.7/0.15/0.15 automatically.

If you to provide training data manually you can use --train_data_path, --test_data_path, and --validation_data_path parameters.

## Calling the OpenAI API to Generate Training Data

To call the OpenAI API, you need to set the `OPENAI_API_KEY` environment variable to your API key.

An example script is located at `src/s2_fos/training/open_ai_prompts.py`. Run it with the following commands:

```bash
poetry shell
poetry run python src/s2_fos/training/open_ai_prompts.py
```

This script reads data from `data/paper_title_abstract_example.json` and writes the results to `data/paper_title_abstract_example_openai.json`.

The OpenAI prompt configuration is defined within `src/s2_fos/training/open_ai_prompts.py`.
