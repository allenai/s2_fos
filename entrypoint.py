import os
import signal
import subprocess
import sys

import click


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

    print("Starting the inference server with {} workers.".format(num_worker_processes))

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
            f"api:initialize_api({batch_size})",
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
    print("Inference server exiting")


"""
Implement other CLI commands as desired using `click`.

```
@cli.command()
def train():
    raise NotImplementedError()
```
"""


if __name__ == "__main__":
    cli()
