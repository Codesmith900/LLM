from fastapi import FastAPI
from pydantic import BaseModel
import spacy
from nltk.stem import PorterStemmer
import uvicorn

app = FastAPI()
nlp = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()

class TextInput(BaseModel):
    text: str

@app.post("/process")
def process_text(data: TextInput):
    doc = nlp(data.text)
    return {
        "tokens": [token.text for token in doc],
        "lemmas": [token.lemma_ for token in doc],
        "stems": [stemmer.stem(token.text) for token in doc],
        "pos": [(token.text, token.pos_) for token in doc],
        "entities": [(ent.text, ent.label_) for ent in doc.ents]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
