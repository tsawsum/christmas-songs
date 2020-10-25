import keras
import os
import numpy as np
import random


def load_data(data_folder, words, valid_punctuation, input_length):
    # Load all text from given files
    all_text = list()
    valid_chars = [chr(k) for k in range(ord('a'), ord('z') + 1)] + valid_punctuation
    for filename in os.listdir(data_folder):
        with open(os.path.join(data_folder, filename), 'r') as f:
            s = ''.join(line for line in f)
            all_text.append(''.join(c for c in s.lower() if c in valid_chars).strip())
    # Split into characters/words
    data = list()
    for text in all_text:
        if words:
            for line in text.split('\n'):
                data[-1] += line.split(' ') + ['\n']
        else:
            data[-1] += [char for char in text] + ['\n']
    # Create mappings
    unique_values = set()
    for values in data:
        unique_values.update(values)
    unique_values.remove('')
    sorted_vals = sorted(unique_values)
    encodings = np.identity(len(sorted_vals))
    val_to_num = {sorted_vals[i]: encodings[i] for i in range(len(unique_values))}
    num_to_val = {v: k for k, v in val_to_num.items()}
    # Format into data set
    input_data, output_data = list(), list()
    shape = len(sorted_vals) * input_length
    for values in data:
        for i in range(input_length, len(values)):
            input_data.append(np.reshape([val_to_num[val] for val in values[i-input_length:i]], shape))
            output_data.append(val_to_num[values[i]])
    return input_data, output_data, num_to_val
    
def train_data(data_folder, save_file, input_length, lstm_size, epochs,
               batch_size, validation_split, valid_punctuation, words=False):
    # Get data
    input_data, output_data, mapping = load_data(data_folder, words, valid_punctuation, input_length)
    mapping_length = len(random.choice(mapping.keys()))
    # Create network
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(lstm_size, input_shape=input_data[0].shape, return_sequences=True))
    model.add(keras.layers.LSTM(lstm_size))
    model.add(keras.layers.Dense(mapping_length, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
