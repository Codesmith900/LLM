from fastapi import FastAPI
from pydantic import BaseModel
from glove_utils import load_glove_embeddings, get_embedding, find_neighbors

app = FastAPI()
embeddings = load_glove_embeddings()

class WordInput(BaseModel):
    word: str

@app.post("/embedding")
def get_word_embedding(input: WordInput):
    vec = get_embedding(input.word, embeddings)
    if vec is None:
        return {"error": "Word not found in GloVe."}
    neighbors = find_neighbors(input.word, embeddings)
    return {"word": input.word, "embedding": vec.tolist(), "neighbors": neighbors}
