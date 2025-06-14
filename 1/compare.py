import spacy
from nltk.stem import PorterStemmer
import pandas as pd

nlp = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()

words = ["running", "flies", "mice", "better", "feet", "studies", "flying", "ate", "happier", "bought"]
doc = nlp(" ".join(words))

data = []

for token in doc:
    stem = stemmer.stem(token.text)
    lemma = token.lemma_
    data.append((token.text, stem, lemma))

df = pd.DataFrame(data, columns=["Word", "Stem (Porter)", "Lemma (spaCy)"])
print(df)
