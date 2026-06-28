import numpy as np
import tensorflow as tf
import random
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Activation
from tensorflow.keras.optimizers import RMSprop

MAX_LENGTH = 10
STEPS = 1
EPOCHS = 20
BATCH_SIZE = 128
TEMPERATURE = 0.5
GENERATE_WORDS = 1000

def load_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def prepare_data(text):
    words = text.lower().split()
    vocab = sorted(list(set(words)))
    w2i = {w: i for i, w in enumerate(vocab)}
    i2w = {i: w for i, w in enumerate(vocab)}
    vocab_size = len(vocab)

    sentences = []
    next_words = []
    for i in range(0, len(words) - MAX_LENGTH, STEPS):
        sentences.append(words[i: i + MAX_LENGTH])
        next_words.append(words[i + MAX_LENGTH])

    X = np.zeros((len(sentences), MAX_LENGTH, vocab_size), dtype=bool)
    y = np.zeros((len(sentences), vocab_size), dtype=bool)

    for i, seq in enumerate(sentences):
        for j, word in enumerate(seq):
            X[i, j, w2i[word]] = 1
        y[i, w2i[next_words[i]]] = 1
    return X, y, vocab_size, w2i, i2w

def sample_preds(preds, temperature):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.multinomial(1, preds, 1).argmax()

def build_model(seq_len, vocab_size):
    model = Sequential()
    model.add(LSTM(128, input_shape=(seq_len, vocab_size)))
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.01))
    return model

def generate_text(model, words, w2i, i2w, seq_len, num_words, temp):
    start_idx = random.randint(0, len(words) - seq_len - 1)
    current_seq = words[start_idx: start_idx + seq_len]
    result = list(current_seq)

    for _ in range(num_words):
        x_pred = np.zeros((1, seq_len, len(w2i)))
        for j, w in enumerate(current_seq):
            if w in w2i:
                x_pred[0, j, w2i[w]] = 1
        preds = model.predict(x_pred, verbose=0)[0]
        next_idx = sample_preds(preds, temp)
        next_word = i2w[next_idx]
        result.append(next_word)
        current_seq = current_seq[1:] + [next_word]
    return ' '.join(result)

def main():
    input_path = 'src/input.txt'
    output_path = 'result/gen.txt'
    text = load_text(input_path)
    X, y, vocab_size, w2i, i2w = prepare_data(text)
    model = build_model(MAX_LENGTH, vocab_size)
    model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)
    words = text.lower().split()
    generated = generate_text(model, words, w2i, i2w, MAX_LENGTH, GENERATE_WORDS, TEMPERATURE)
    os.makedirs('result', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(generated)

if __name__ == '__main__':
    main()
