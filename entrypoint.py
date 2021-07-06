import os
import signal
import subprocess
import sys


def exit_process(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass

    sys.exit(0)


def main():
    if sys.argv[1] == "serve":
        uvicorn = subprocess.Popen(
            [
                "uvicorn",
                "--host",
                "0.0.0.0",
                "api:api"
            ]

        )

        signal.signal(signal.SIGTERM, lambda a, b: exit_process(uvicorn.pid))

        while True:
            pid, _ = os.wait()
            if pid == uvicorn.pid:
                # Uvicorn is exited and we should exit this parent process as well
                break

        print("Inference server exiting")

        sys.exit(0)

    else:
        print(f"Unrecognized command '{sys.argv[1]}'")


if __name__ == '__main__':
    main()
