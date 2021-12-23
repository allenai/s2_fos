FROM python:3.8-slim-buster

###########################################################
###### MODEL SPECIFIC SYSTEM DEPS AND SETUP GO HERE #######
###########################################################

# have at it


###########################################################
#### S2AGEMAKER REQS BELOW THIS POINT -- DO NOT CHANGE ####
###########################################################

RUN apt-get update && apt-get install nginx iproute2 gcc g++ -y
WORKDIR /opt/ml/code

COPY PROJECT_NAME.txt .
COPY requirements.txt .
COPY setup.py .

RUN pip install -r requirements.txt
RUN pip install --upgrade .[dev]

COPY . .

ENV ARTIFACTS_DIR /opt/ml/model
ENV BATCH_SIZE 1
ENV MODEL_SERVER_TIMEOUT 60
ENV NUM_WORKERS 1
ENV PYTHONUNBUFFERED 1

ENTRYPOINT ["python3", "entrypoint.py"]
CMD ["serve"]

# For training and eval routines
RUN mkdir -p /opt/ml/input
RUN mkdir -p /opt/ml/output

ENV INPUT_DATA_DIR /opt/ml/input/data
ENV INPUT_CONFIG_DIR /opt/ml/input/config
ENV OUTPUT_DATA_DIR /opt/ml/output

ENV CHANNEL_NAME $CHANNEL_NAME
ENV MODEL_VERSION $MODEL_VERSION
ENV HYPERPARAMETERS_FILE $HYPERPARAMETERS_FILE
