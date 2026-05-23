from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}

@app.post("/predict")
def predict(data: list):
    arr = np.array(data).reshape(1, -1)
    arr = scaler.transform(arr)
    prediction = model.predict(arr)[0]
    
    return {
        "prediction": int(prediction),
        "result": "Churn" if prediction == 1 else "No Churn"
    }