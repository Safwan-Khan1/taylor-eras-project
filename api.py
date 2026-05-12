from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import json
import os
from src.model import load_model

app = FastAPI(title="Taylor Swift Eras API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = load_model('model.pkl')

class PredictionRequest(BaseModel):
    # Mocking what the user might input
    lyrics: str
    acousticness: float
    danceability: float
    energy: float
    instrumentalness: float
    liveness: float
    loudness: float
    speechiness: float
    valence: float
    tempo: float

@app.get("/")
def read_root():
    return {"message": "Welcome to Taylor Swift Eras Architecture API"}

@app.get("/model_info")
def get_model_info():
    if not os.path.exists('model_data.json'):
        raise HTTPException(status_code=404, detail="Model data not found. Please train the model first.")
    with open('model_data.json', 'r') as f:
        data = json.load(f)
    return data

@app.post("/predict")
def predict_era(req: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")
    
    # Format Audio Vector
    X_audio = np.array([[
        req.acousticness, req.danceability, req.energy, 
        req.instrumentalness, req.liveness, req.loudness, 
        req.speechiness, req.valence, req.tempo
    ]])
    
    # Format Text Vector
    X_text = [req.lyrics]
    
    prediction = model.predict(X_audio, X_text)[0]
    probabilities = model.predict_proba(X_audio, X_text)[0]
    
    # Create probability dictionary
    prob_dict = {str(c): float(p) for c, p in zip(model.classes_, probabilities)}
    
    return {
        "prediction": prediction,
        "probabilities": prob_dict
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
