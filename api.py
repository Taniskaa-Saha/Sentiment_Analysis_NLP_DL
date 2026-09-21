from api import FastAPI, StaticFiles
from pydantic import BaseModel, Field
from keras.models import load_model
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
import re
import pickle

app=FastAPI()

@app.get("/")
def greet():
    return { "Hello, World!"}


#model path
model_path="ARTIFACTS/BiGRU_model.keras"

#Tokenizer path
tokenizer_path="ARTIFACTS/tokenizer.pkl"

#MaxSequence Length
max_sequence_length = 50

#emotion classes
emotion_classes = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']

#emotion emojis
emotion_emojis = {
    'anger': '😠',
    'fear': '😨',
    'joy': '😄',
    'love': '❤️',
    'sadness': '😢',
    'surprise': '😲'
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
        dl_model["Tokenizer "] = pickle.load(file)
    print("Model and tokenizer loaded.")

    yield #pause, model Is loaded and softer is running at this point model waits for request.
    dl_model.clear()  #clear the model and tokenizer from memory when server shuts down

#mount static files to  FAST api
#enable CORS (Cross-Origin Resource Sharing) to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount('/static', StaticFiles(directory='static'), name='static')

