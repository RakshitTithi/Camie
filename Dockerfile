FROM python:3.8.6-buster

# Base
COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Specify where are the files for this
#RUN mkdir /project
WORKDIR /project

# Copy the files to the WORKDIR
COPY * .
COPY taxifare ./taxifare
COPY taxifare.egg-info ./taxifare.egg-info

ENV LOCAL_DATA_PATH=/project/lewagon/mlops/data
ENV LOCAL_REGISTRY_PATH=/project/lewagon/mlops/training_outputs

ENV DATASET_SIZE=10k
ENV VALIDATION_DATASET_SIZE=10k
ENV CHUNK_SIZE=6000

ENV DATA_SOURCE=local
ENV MODEL_TARGET=local

# GCP Project
ENV PROJECT=lewagon-366815
ENV REGION=europe-west3

# Cloud Storage
ENV BUCKET_NAME=le-wagon-bucket-opitz
ENV BLOB_LOCATION=data

# BigQuery (multi region must be EU since le wagon cloud storage public datasets are in EU)
ENV MULTI_REGION=EU
ENV DATASET=taxifare_dataset

# Compute Engine
ENV INSTANCE=taxi-instance


# Model Lifecycle  
# user.github_nickname -> leonardoopitz.leonardoopitz ???
#MLFLOW_TRACKING_URI=https://mlflow.lewagon.ai
#MLFLOW_EXPERIMENT=taxifare_experiment_<user.github_nickname>
#MLFLOW_MODEL_NAME=taxifare_<user.github_nickname>

ENV PREFECT_BACKEND=development
ENV PREFECT_FLOW_NAME=taxifare_lifecycle_<user.github_nickname>
ENV PREFECT_LOG_LEVEL=WARNING

# API
ENV MODEL_TARGET=mlflow
ENV MLFLOW_TRACKING_URI=https://mlflow.lewagon.ai
ENV MLFLOW_EXPERIMENT=taxifare_experiment_krokrob
ENV MLFLOW_MODEL_NAME=taxifare_krokrob



EXPOSE 8000

# CMD make run_api 
# CMD uvicorn api.simple:app --host 0.0.0.0 --port 8000  

# ENTRYPOINT ["/project/make run_api", "-D", "FOREGROUND"]   
