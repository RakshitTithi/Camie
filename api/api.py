from datetime import datetime
from unittest import result
from urllib import response
from random import *
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# from taxifare.interface.main import pred
from taxifare.ml_logic.preprocessor import preprocess_features
from taxifare.ml_logic.registry import load_model
import pandas as pd
from datetime import datetime
import math


app = FastAPI()
#app.state.model = load_model() # MLFLOW_MODEL_NAME

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?pickup_datetime=2012-10-0
@app.get("/send_photo")
def send_photo(camera_id, secret, photo):

    # Security
    our_secret = get_secret_for(camera_id)
    if our_secret != secret:
        return '{"msg": "sorry... but you dont have access"}'

    # Saves the photo and info
    saves_photo(photo)

    # ML Process the image
    num_people = math.rand(1,10)

    # Saves the info
    saves_info(camera_id, photo, num_people)

    # 3rd returns the results
    response = {'num_people': num_people}
    return response


@app.get("/log_camera")
def log_camera(camera_id, password):

    # 1st generates the secret for this camera
    initial_password = "TheBestTeam!746"
    if (password == initial_password):
        secret = generate_secrete()

    # 2nd saves the info

    # 3rd returns the secret
    response = {'secret': secret}
    return response


@app.get("/get_info")
def get_info(dashboard_secret):

    # 1st generates the secret for this camera
    saved_dashboard_secret = "TheBestTeam!746"
    if dashboard_secret != saved_dashboard_secret():
        return '{"msg": "sorry... but you dont have access"}'

    # 2nd retries the info
    results = []
    infos = []
    for info in infos:
        line = ""
        results.append(line)

    # 3rd returns the results
    # There is no need for a return after 1st response. Return at end
    response = {"code": 0, 'result': results}
    if len(results) == 0:
        response = {"code": 1, 'result': ""}
    return response






@app.get("/")
def root():
    response = {'greeting': 'Hello'}
    return response




def generate_secrete():
    return randint(1, 99999)

def get_dashbord_password():
    return randint(1, 99999)
