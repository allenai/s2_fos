import json
import logging
import logging.config
import os
import signal
import subprocess
import sys

import click

from model.eval import EvalSettings, generate_metrics
from model.predictor import Predictor, PredictorConfig
from model.training import build_and_train_model, TrainingConfig
from model.utils import load_labeled_data, save_model


dir = os.path.dirname(os.path.realpath(__file__))
logging.config.fileConfig(os.path.join(dir, "server", "logging.conf"))
logger = logging.getLogger(__name__)


def _sigterm_handler(nginx_pid, gunicorn_pid):
    try:
        os.kill(nginx_pid, signal.SIGQUIT)
    except OSError:
        pass
    try:
        os.kill(gunicorn_pid, signal.SIGTERM)
    except OSError:
        pass

    sys.exit(0)


@click.group()
def cli():
    pass


@cli.command()
def serve():
    model_server_timeout = int(os.environ.get("MODEL_SERVER_TIMEOUT", 60))
    num_worker_processes = int(os.getenv("NUM_WORKERS", 1))
    batch_size = int(os.getenv("BATCH_SIZE", 1))

    # getting model artifacts from S3
    model_s3_path = os.getenv("MODEL_S3_PATH")
    if model_s3_path:
        target_path = os.environ["ARTIFACTS_DIR"]
        logging.info(f"Getting model artifacts from {model_s3_path} to {target_path}")
        get_model_artifacts(model_s3_path, target_path)
    else:
        logging.info(f"Getting model artifact locally.")

    logger.info(
        "Starting the inference server with {} workers.".format(num_worker_processes)
    )

    # link the log streams to stdout/err so they will be logged to the container logs
    subprocess.check_call(["ln", "-sf", "/dev/stdout", "/var/log/nginx/access.log"])
    subprocess.check_call(["ln", "-sf", "/dev/stderr", "/var/log/nginx/error.log"])

    nginx = subprocess.Popen(["nginx", "-c", "/opt/ml/code/nginx.conf"])
    gunicorn = subprocess.Popen(
        [
            "gunicorn",
            "--timeout",
            str(model_server_timeout),
            "-k",
            "uvicorn.workers.UvicornWorker",
            "-b",
            "unix:/tmp/gunicorn.sock",
            "-w",
            str(num_worker_processes),
            "--log-config",
            "./server/logging.conf",
            f"server.api:initialize_api({batch_size})",
        ]
    )

    signal.signal(
        signal.SIGTERM, lambda a, b: _sigterm_handler(nginx.pid, gunicorn.pid)
    )

    # If either subprocess exits, so do we.
    pids = set([nginx.pid, gunicorn.pid])
    while True:
        pid, _ = os.wait()
        if pid in pids:
            break

    _sigterm_handler(nginx.pid, gunicorn.pid)
    logger.info("Inference server exiting")

def get_model_artifacts(model_s3_path: str, target_path: str):
    subprocess.check_call(
        [
            "aws",
            "s3",
            "sync",
            model_s3_path,
            target_path
        ]
    )

@cli.command()
def train():
    logger.info("BEGINNING TRAINING ROUTINE")

    training_config = TrainingConfig()
    hyperparameters = training_config.load_hyperparameters()
    training_examples = load_labeled_data(training_config.training_data_dir())

    logger.info(
        f"""
        TRAINING WITH
        data channel: {training_config.channel_name}
        hyperparameters: {hyperparameters.json()}
    """
    )
    trained_model = build_and_train_model(training_examples, hyperparameters)

    logger.info("MODEL TRAINING COMPLETE, SAVING TO DISK")
    logger.info(f"MODEL VERSION: {training_config.model_version}")
    save_model(training_config.target_artifacts_dir(), hyperparameters, trained_model)
    logger.info("MODEL SAVED, EXITING")


@cli.command()
def evaluate():
    logger.info("BEGINNING EVAL ROUTINE")

    eval_settings = EvalSettings()
    eval_examples = load_labeled_data(eval_settings.eval_data_dir())

    predictor_config = PredictorConfig()
    predictor = Predictor(predictor_config)

    logger.info(
        f"""
        RUNNING EVAL FOR
        model version: {eval_settings.model_version}
        dataset: {eval_settings.channel_name}
    """
    )

    predictions = predictor.predict_batch([ex.instance for ex in eval_examples])
    metrics = generate_metrics(eval_examples, predictions)

    metrics_dir = eval_settings.metrics_output_dir()

    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    logging.info(
        f"""
        METRICS:
        {metrics}
        """
    )

    with open(os.path.join(metrics_dir, "metrics.json"), "w") as f:
        f.write(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    cli()
