import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data(file_path, num_samples=5000):
    df = pd.read_csv(file_path, encoding="ISO-8859-1")[['Article', 'Heading']].dropna()
    return df['Article'][:num_samples].tolist(), df['Heading'][:num_samples].tolist()

def tokenize_texts(texts, num_words=10000, max_len=100):
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded, tokenizer
