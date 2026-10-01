# EmotionSense: Sentiment Analysis with BiGRU

EmotionSense is a deep learning-based text classification application built using **TensorFlow/Keras**. A **Bidirectional GRU (BiGRU)** model predicts emotions from text, while a **FastAPI** backend provides prediction and health-check endpoints. The application is deployed on **Render** with a responsive HTML, CSS, and JavaScript frontend.

**Live Demo:** https://sentiment-analysis-nlp-dl.onrender.com

---

## Features

* BiGRU-based deep learning model for emotion classification
* FastAPI backend with `/predict` and `/health` endpoints
* Responsive frontend using HTML, CSS, and JavaScript
* Real-time prediction with confidence score
* API health check
* Deployed on Render

---

## Tech Stack

| Layer       | Technology                     |
| ----------- | ------------------------------ |
| Programming | Python                         |
| Model       | TensorFlow / Keras (BiGRU)     |
| NLP         | Tokenization, sequence padding |
| Backend     | FastAPI, Uvicorn               |
| Frontend    | HTML, CSS, JavaScript          |
| Deployment  | Render                         |

---

## Project Structure

```text
Sentiment_Analysis_NLP_DL/
│
├── api.py
├── requirements.txt
├── runtime.txt
│
├── ARTIFACTS/
│   ├── BiGRU_model.keras
│   └── tokenizer.pkl
│
└── static/
    ├── index.html
    ├── style.css
    └── script.js
```

---

## Model Overview

* **Architecture:** Embedding → Bidirectional GRU → Dense → Output
* **Task:** Multi-class emotion classification
* **Classes:** Sadness, Anger, Love, Surprise, Fear, Joy
* **Preprocessing:** Tokenization and sequence padding
* **Maximum sequence length:** 50 tokens
* **Maximum vocabulary size:** 10,000 tokens
* **Best Model:** BiGRU
* **Accuracy:** 92.30%

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Taniskaa-Saha/Sentiment_Analysis_NLP_DL.git
cd Sentiment_Analysis_NLP_DL
```

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m uvicorn api:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

---

## API Reference

### `GET /health`

Checks whether the API is running.

**Response:**

```json
{
  "status": "ok"
}
```

### `POST /predict`

Predicts the emotion of the provided text.

**Request:**

```json
{
  "text": "I am feeling very happy today!"
}
```

**Response:**

```json
{
  "predicted_emotion": "joy",
  "confidence": 0.92
}
```

---

## Deployment

The application is deployed on **Render**.

**Build Command:**

```bash
pip install -r requirements.txt
```

**Start Command:**

```bash
uvicorn api:app --host 0.0.0.0 --port $PORT
```

**Health Check Path:**

```text
/health
```

The frontend uses relative API paths, allowing the application to work without hard-coded localhost URLs.

---

## Notes

* The trained BiGRU model and tokenizer are stored in the `ARTIFACTS` directory.
* The model expects sequences with a maximum length of 50 tokens.
* TensorFlow and other dependencies should remain compatible with the saved model.
* Render's free instance may take some time to respond after inactivity.

---

## Future Improvements

* Improve handling of negation and contextual language
* Experiment with LSTM and Transformer-based models
* Add batch prediction
* Improve model performance across emotion classes
* Add Docker-based deployment
* Add model monitoring and experiment tracking with MLflow

---

## Author

**Taniskaa Saha**

* GitHub: https://github.com/Taniskaa-Saha
* LinkedIn: https://www.linkedin.com/in/taniskaa-saha-b70885272/
