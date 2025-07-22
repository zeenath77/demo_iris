from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np


from joblib import load
ml = load("C:/Users/zeena/OneDrive/Desktop/apiproject/apiproject_iris.pkl")

# Load the saved model

spec_cls=["setosa","versicolor","virginica"]

#creating a class
class iris_inp(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    #based on the above inputs it will predict which kind of flower it is

app=FastAPI()

@app.get('/')
def read_load():
    return "Welcome to the iris flower sepcies prediction"


@app.post('/prd')
def prediction(data: iris_inp):
    fea = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
    prd_out = ml.predict(fea)
    spe_name = spec_cls[int(prd_out[0])]
    return {"Predicted_Species": spe_name}


