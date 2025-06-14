from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Dropout

def build_seq2seq(vocab_size_enc, vocab_size_dec, embedding_dim=128, units=128, max_len_input=100, max_len_output=20):
    # Encoder
    encoder_inputs = Input(shape=(max_len_input,))
    enc_emb = Embedding(vocab_size_enc, embedding_dim, mask_zero=True)(encoder_inputs)
    encoder_lstm, state_h, state_c = LSTM(units, return_state=True)(enc_emb)

    # Decoder
    decoder_inputs = Input(shape=(max_len_output,))
    dec_emb = Embedding(vocab_size_dec, embedding_dim, mask_zero=True)(decoder_inputs)
    decoder_lstm, _, _ = LSTM(units, return_sequences=True, return_state=True)(dec_emb, initial_state=[state_h, state_c])
    dropout = Dropout(0.5)(decoder_lstm)
    outputs = Dense(vocab_size_dec, activation='softmax')(dropout)

    model = Model([encoder_inputs, decoder_inputs], outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
