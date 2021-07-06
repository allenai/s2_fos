FROM python:3.8

RUN mkdir -p /opt/ml/code
WORKDIR /opt/ml/code

COPY requirements.txt .
COPY setup.py .

RUN pip install -r requirements.txt
RUN pip install --upgrade .[dev]

COPY . .

ENV PATH $PATH:/opt/ml/code

ENTRYPOINT ["python3", "entrypoint.py"]
CMD ["serve"]


