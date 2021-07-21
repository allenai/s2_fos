#!/usr/bin/env bash

set -e

export TAG="${BUILD_NUMBER:-$(date +%s)}"
export TTY_DEVICE=0

echo "=====BUILDING IMAGE====="
make build-image

echo "=====RUNNING MYPY====="
make mypy

echo "=====RUNNING TESTS====="
make unit

echo "=====FORMATTING====="
make format
