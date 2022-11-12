from datetime import datetime
import imp
from unittest import result
from urllib import response
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from taxifare.interface.main import pred
from taxifare.ml_logic.preprocessor import preprocess_features
from taxifare.ml_logic.registry import load_model
import pandas as pd
from datetime import datetime


app = FastAPI()
app.state.model = load_model() # MLFLOW_MODEL_NAME

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?pickup_datetime=2012-10-06 12:10:20&pickup_longitude=40.7614327
# &pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2
@app.get("/predict")
def predict(pickup_datetime: datetime,  # 2013-07-06 17:18:00
            pickup_longitude: float,    # -73.950655
            pickup_latitude: float,     # 40.783282
            dropoff_longitude: float,   # -73.984365
            dropoff_latitude: float,    # 40.769802
            passenger_count: int):      # 1
    """
    we use type hinting to indicate the data types expected
    for the parameters of the function
    FastAPI uses this information in order to hand errors
    to the developpers providing incompatible parameters
    FastAPI also provides variables of the expected data type to use
    without type hinting we need to manually convert
    the parameters of the functions which are all received as strings
    """


    #today = datetime.today()
    #pickup_datetime = today

    print(f"pickup_datetime: {pickup_datetime}")
    print(f"pickup_longitude: {pickup_longitude}")
    print(f"pickup_latitude: {pickup_latitude}")
    print(f"dropoff_longitude: {dropoff_longitude}")
    print(f"dropoff_latitude: {dropoff_latitude}")
    print(f"passenger_count: {passenger_count}")

    print("----- LO: NOW CONVERTING ---------------------------")

    df = pd.DataFrame(
        {
            'key': pickup_datetime,
            'pickup_datetime': pickup_datetime,
            'pickup_longitude': pickup_longitude,
            'pickup_latitude': pickup_latitude,
            'dropoff_longitude': dropoff_longitude,
            'dropoff_latitude': dropoff_latitude,
            'passenger_count': passenger_count
        },
        index=[0])


    print("----- LO: NOW PREDICTING ---------------------------")
    X_processed = preprocess_features(df)
    y_pred = app.state.model.predict(X_processed)
    print("----- LO: BACK FROM PREDICTING ---------------------------")

    response = {'fare_amount': round(float(y_pred[0]),2)}
    return response


@app.get("/")
def root():
    response = {'greeting': 'Hello'}
    return response

