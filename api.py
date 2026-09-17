from fastapi import FastAPI
from pydantic import BaseModel, Field

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
import re
def preprocess_text(text:str)-> str:
    #lowercase the text
    text=text.lower()
    text=re.sub(r"'", "", text)
    text=re.sub(r"[^a-z0-9\s]", " ", text)
    text=re.sub(r"\s+", " ", text).strip()
    return text

class TextRequest(BaseModel):
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