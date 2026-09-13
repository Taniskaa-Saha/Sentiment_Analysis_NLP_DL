from fastapi import FastAPI

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