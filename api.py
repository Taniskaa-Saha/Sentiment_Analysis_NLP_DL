from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from keras.models import load_model
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from pathlib import Path
import numpy as np
import re
import pickle


#model path
BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "ARTIFACTS" / "BiGRU_model.keras"

#Tokenizer path
tokenizer_path = BASE_DIR / "ARTIFACTS" / "tokenizer.pkl"

#MaxSequence Length
max_sequence_length = 50

#emotion classes
emotion_classes = ['sadness', 'anger', 'love', 'surprise', 'fear', 'joy']

#emotion emojis
emotion_emojis = {
    'sadness': '😢',
    'anger': '😠',
    'love': '❤️',
    'surprise': '😲',
    'fear': '😨',
    'joy': '😄',
}

#preprocess upcomming text
def preprocess_text(text:str)-> str:
    #lowercase the text
    text=text.lower()
    text=re.sub(r"'", "", text)
    text=re.sub(r"[^a-z0-9\s]", " ", text)
    text=re.sub(r"\s+", " ", text).strip()
    return text

class TextInput(BaseModel):
    text: str = Field(..., 
                      min_length=1,
                      max_length=2000,
                      description="The text to analyze for sentiment",
                      json_schema_extra={
                          "example": "I am so happy today!"}
                          )

class PredictionResponse(BaseModel):
    text: str
    predicted_emotion: str
    confidence: float
    all_probabilities: dict[str, float]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


#Model loading and Lifespan Management
#Load the model and tokenizer once the server stats up.

dl_model = {}
async def lifespan(app:FastAPI):
    print("Loading model and tokenizer...")
    #global model, tokenizer
    dl_model["BiGRU"] = load_model(model_path)   #BiGRU model
    with open(tokenizer_path, "rb") as file:
        dl_model["Tokenizer"] = pickle.load(file)
    print("Model and tokenizer loaded.")

    yield #pause, model Is loaded and softer is running at this point model waits for request.
    dl_model.clear()  #clear the model and tokenizer from memory when server shuts down

#mount static files to  FAST api
app=FastAPI(
    lifespan=lifespan,
)

#enable CORS (Cross-Origin Resource Sharing) to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#app.mount('/static', StaticFiles(directory='static'), name='static')
app.mount('/static', StaticFiles(directory=BASE_DIR / "Static"), name='static')

#API endpoints
@app.get('/', include_in_schema=False)
def server_ui():
    return FileResponse(BASE_DIR / "Static" / "index.html")

@app.get('/health', response_model=HealthResponse)
def health_check():
    return HealthResponse(status="server is running", model_loaded=bool(dl_model))

@app.post('/predict', response_model=PredictionResponse)
def predict_emotion(input_data: TextInput):
    BiGRU_model = dl_model.get("BiGRU")
    tokenizer = dl_model.get("Tokenizer")

    if BiGRU_model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model or tokenizer not loaded. Please try again later.")

    cleaned_text = preprocess_text(input_data.text)
    tokenized_text = tokenizer.texts_to_sequences([cleaned_text])
    print(cleaned_text)
    padded_text = pad_sequences(
        tokenized_text, 
        maxlen=max_sequence_length, 
        padding='post', 
        truncating='post'
        )

    probabilities = BiGRU_model.predict(padded_text)[0]
    top_emotion_index = int(np.argmax(probabilities))
    all_probabilities = {
        label: float(prob) 
        for prob, label in zip(probabilities, emotion_classes)
    }

    return PredictionResponse(
        text=input_data.text,
        predicted_emotion=emotion_classes[top_emotion_index],
        confidence=float(probabilities[top_emotion_index]),
        all_probabilities=all_probabilities
    )