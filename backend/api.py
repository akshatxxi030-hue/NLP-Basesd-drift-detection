from fastapi import FastAPI
import pandas as pd
import numpy as np
from backend.drift_calculation import drift_pipeline

app=FastAPI()

@app.get('/')
def home():
    return{'message':'NLP based drift detection'}

@app.get('/health')
def health_check():
    return{
        'status':'OK',
        'API last updated':'28/01/26'

    }
@app.get('/predict')
def predict():
    return drift_pipeline()

