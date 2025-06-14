from preprocess import load_data, tokenize_texts
from model import build_seq2seq
import numpy as np

# Load and preprocess data
articles, summaries = load_data("data/Articles.csv", num_samples=5000)

max_len_article = 100
max_len_summary = 20

article_padded, article_tokenizer = tokenize_texts(articles, max_len=max_len_article)
summary_padded, summary_tokenizer = tokenize_texts(summaries, max_len=max_len_summary)

# Prepare decoder input/output
decoder_input = summary_padded[:, :-1]
decoder_output = summary_padded[:, 1:]
decoder_output = np.expand_dims(decoder_output, -1)

# Build model
vocab_size_encoder = len(article_tokenizer.word_index) + 1
vocab_size_decoder = len(summary_tokenizer.word_index) + 1

model = build_seq2seq(vocab_size_encoder, vocab_size_decoder,
                      max_len_input=max_len_article, max_len_output=max_len_summary - 1)

# Train without early stopping
model.fit([article_padded, decoder_input], decoder_output,
          batch_size=64, epochs=30, validation_split=0.1)
