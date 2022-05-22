FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update \
    && apt-get install -y wget \
    && apt-get install -y unzip

COPY . .

# Get Stanford data, unzip it and prepare it
RUN wget http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip
RUN unzip stanfordSentimentTreebank.zip
RUN python -m scripts.prepare_data

ENV WANDB_API_KEY=29786b4d3965893d978b29e824dade840a09c74e

ENTRYPOINT ["python", "-m", "scripts.fit_and_evaluate"]