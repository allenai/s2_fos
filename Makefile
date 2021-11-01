PROJECT_NAME=$(shell cat PROJECT_NAME.txt)

ifdef TAG
	OPTIONAL_TAG = -$(TAG)
endif

IMAGE_NAME=$(PROJECT_NAME)$(OPTIONAL_TAG)

build-image:
	docker build -t $(IMAGE_NAME) .

serve: build-image
	docker run --rm \
		-p 8080:8080 \
		--env-file $(shell pwd)/docker.env \
		-v $(shell pwd)/artifacts:/opt/ml/model:ro \
		$(IMAGE_NAME) serve

train: build-image
	docker run \
		--env-file $(shell pwd)/docker.env \
		-v $(shell pwd)/artifacts:/opt/ml/model:rw \
		-v $(shell pwd)/input:/opt/ml/input:ro \
		$(IMAGE_NAME) train

eval: build-image
	docker run \
		--env-file $(shell pwd)/docker.env \
		-v $(shell pwd)/artifacts:/opt/ml/model:ro \
		-v $(shell pwd)/input:/opt/ml/input:ro \
		-v $(shell pwd)/output:/opt/ml/output:rw \
		$(IMAGE_NAME) evaluate

type-check: build-image
	docker run --rm --entrypoint /bin/bash $(IMAGE_NAME) -c 'mypy .'

format-check: build-image
	docker run --rm --entrypoint /bin/bash $(IMAGE_NAME) -c 'black --check .'

format: build-image
	docker run --rm -v $(shell pwd):/opt/ml/code --entrypoint bash $(IMAGE_NAME) -c 'black .'

unit: build-image
	docker run --rm --entrypoint /bin/bash $(IMAGE_NAME) -c 'pytest tests'
