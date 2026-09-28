from fastapi import FastAPI
from app.embedding_model import calculate_embedding

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Word Embedding API is running"}


@app.get("/embedding/{word}")
def get_embedding(word: str):
    embedding = calculate_embedding(word)

    return {
        "word": word,
        "embedding": embedding.tolist()
    }