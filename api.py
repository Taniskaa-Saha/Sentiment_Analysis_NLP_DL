import pickle
import re
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from keras.models import load_model
from pydantic import BaseModel, Field
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer  # noqa: F401  (needed so pickle can rebuild the tokenizer)


# ---------------------------------------------------------------------------
# Paths
# NOTE: Render runs Linux, where folder names are case-sensitive.
# These names must match the folders in your GitHub repo EXACTLY.
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "Artifacts"   # training code saves to "Artifacts"
STATIC_DIR = BASE_DIR / "Static"

MODEL_PATH = ARTIFACTS_DIR / "BiGRU_model.keras"
TOKENIZER_PATH = ARTIFACTS_DIR / "tokenizer.pkl"

# Must match the value used during training
MAX_SEQUENCE_LENGTH = 50

# Must match the training label_mapping order:
# sadness=0, anger=1, love=2, surprise=3, fear=4, joy=5
EMOTION_CLASSES = ["sadness", "anger", "love", "surprise", "fear", "joy"]


# ---------------------------------------------------------------------------
# Text preprocessing (identical to clean_text in the training code)
# ---------------------------------------------------------------------------
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.replace("cannot", "can not")
    text = re.sub(r"\bcan'?t\b", "can not", text)
    text = re.sub(r"\bwon'?t\b", "will not", text)
    text = re.sub(
        r"\b(did|do|does|is|was|are|were|have|has|had|could|would|should)n'?t\b",
        r"\1 not",
        text,
    )
    text = text.replace("'", "")
    return text


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class TextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The text to analyze for emotion",
        json_schema_extra={"example": "I am so happy today!"},
    )


class PredictionResponse(BaseModel):
    text: str
    predicted_emotion: str
    confidence: float
    all_probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# ---------------------------------------------------------------------------
# Model loading and lifespan management
# The model and tokenizer are loaded once when the server starts.
# ---------------------------------------------------------------------------
ml_assets: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model and tokenizer...")
    ml_assets["model"] = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as file:
        ml_assets["tokenizer"] = pickle.load(file)
    print("Model and tokenizer loaded.")

    yield  # server runs and handles requests here

    ml_assets.clear()  # free memory on shutdown


app = FastAPI(title="Emotion Detection API", lifespan=lifespan)

# CORS: no cookies/auth are used, so credentials are disabled.
# (allow_origins=["*"] together with allow_credentials=True is invalid.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the HTML/CSS/JS frontend
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def server_ui():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="server is running",
        model_loaded="model" in ml_assets and "tokenizer" in ml_assets,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_emotion(input_data: TextInput):
    model = ml_assets.get("model")
    tokenizer = ml_assets.get("tokenizer")

    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model or tokenizer not loaded. Please try again later.",
        )

    cleaned_text = preprocess_text(input_data.text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(
        sequence,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
    )

    probabilities = model.predict(padded, verbose=0)[0]
    top_index = int(np.argmax(probabilities))

    all_probabilities = {
        label: float(prob) for label, prob in zip(EMOTION_CLASSES, probabilities)
    }

    return PredictionResponse(
        text=input_data.text,
        predicted_emotion=EMOTION_CLASSES[top_index],
        confidence=float(probabilities[top_index]),
        all_probabilities=all_probabilities,
    )
