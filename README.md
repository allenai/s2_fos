# s2-fos

Model code for Semantic Scholar's paper Field of Study classifier.

Uses a multi-label SVM classifier over paper title and abstract text,
embedded as character-level ngrams.

## Training

To use this project's training routine, you must provide a set of labeled
training data, and a hyperparameters config file.

### Training data

Make a new a directory: `input/data/<TRAINING_DATASET_NAME>`. 

Format your labeled examples in the shape of `model.example.Example`
in one or more `JSONL` files under that directory.

Separately, create a hyperparameters JSON config file: `input/config/<YOUR_HYPERPARAMS>`.
Follow the format in `model.hyperparameters.ModelHyperparameters`.

Finally, to invoke training:

```bash
cd <project_root>

CHANNEL_NAME=<TRAINING_DATASET_NAME> \
  HYPERPARAMETERS_FILE=<YOUR_HYPERPARAMS> \
  MODEL_VERSION=<SOME_VERSION_IDENTIFIER> \
  make train
```

This will train your model and output artifacts at: `artifacts/<SOME_VERSION_IDENTIFIER>`.
It will also save the hyperparameters used at training for later reference.

Please note: if no arguments are provided, training data will be loaded from:
`input/data/training`, hyperparams from `input/config/hyperparameters.json` and
artifacts will be saved directly to `artifacts/`.

## Evaluation

You can run a trained model against an evaluation or test dataset in a similar vein.

Make a new directory: `input/data/<EVALUATION_DATASET_NAME>`. Add labeled example files
as with training.

Invoke evaluation with:

```bash
cd <project_root>

CHANNEL_NAME=<EVALUATION_DATASET_NAME> \
  MODEL_VERSION=<SOME_MODEL_IDENTIFIER> \
  make evaluate
```

This will produce predictions using the model artifacts at `artifacts/<SOME_MODEL_IDENTIFIER>`
against the labeled dataset in `input/data/<EVALUATION_DATASET_NAME>`.

Evaluation results will be stored to <TODO>.

Please note: if no `MODEL_VERSION` is provided, artifacts will be loaded directly from
`artifacts/`.

## Serving

To run a trained model as an HTTP callable, prepare your artifacts and run:

```bash
cd <project_root>

MODEL_VERSION=<SOME_MODEL_IDENTIFIER> make serve
```

If no `MODEL_VERSION` is provided, artifacts will be loaded directly from `artifacts/`.
