#!/usr/bin/env bash

set -euo pipefail
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export PROJECT_NAME=$(cat PROJECT_NAME.txt)
export TAG="${BUILD_NUMBER:-$(date +%s)}"
export TTY_DEVICE=0

echo "=====BUILDING IMAGE====="
make build-image

echo "=====RUNNING TYPE CHECK====="
make type-check

echo "=====RUNNING FORMAT CHECK====="
make format-check

echo "=====RUNNING UNIT TESTS====="
make unit

echo "=====RUNNING INTEGRATION TESTS====="
IMAGE=$PROJECT_NAME-$TAG  # matches Makefile
docker run -d --rm --name "$IMAGE" -p 8080:8080 --env-file docker.env \
  -v "$(pwd)"/artifacts:/opt/ml/model:ro "$IMAGE" serve

cleanup() {
  trap - 0
  set +e
  docker stop "$IMAGE"
}
trap cleanup 0

docker run --rm --env-file docker.env --entrypoint bash \
  -v "$(pwd)"/artifacts:/opt/ml/model:ro "$IMAGE" -c 'pytest integration_tests'
