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
            s = ''.join(line.strip() + '\n' for line in f)
            all_text.append(''.join(c for c in s.lower() if c in valid_chars).strip())
    # Split into characters/words
    data = list()
    for text in all_text:
        data.append(list())
        if words:
            for line in text.split('\n'):
                data[-1] += line.split(' ') + ['\n']
        else:
            data[-1] += [char for char in text] + ['\n']
    # Create mappings
    unique_values = set()
    for values in data:
        unique_values.update(values)
    sorted_vals = sorted(unique_values)
    encodings = np.identity(len(sorted_vals))
    val_to_num = {sorted_vals[i]: encodings[i] for i in range(len(unique_values))}
    # Format into data set
    input_data, output_data = list(), list()
    for values in data:
        for i in range(input_length, len(values)):
            input_data.append(np.array([val_to_num[val] for val in values[i-input_length:i]]))
            output_data.append(val_to_num[values[i]])
    return np.array(input_data), np.array(output_data), sorted_vals


def train_data(data_folder, save_file, input_length, lstm_size, epochs,
               batch_size, validation_split, valid_punctuation, words=False):
    # Get data
    input_data, output_data, mapping = load_data(data_folder, words, valid_punctuation, input_length)
    # Create model file
    i = 1
    while os.path.isfile(os.path.join(os.getcwd(), 'model_' + str(i) + '.h5')):
        i += 1
    model_file = os.path.join(os.getcwd(), 'model_' + str(i) + '.h5')
    with open(save_file, 'w+') as f:
        f.write(str(input_length) + '\n')
        f.write(str(model_file) + '\n')
        f.write(';#;'.join(mapping))
    # Create network
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(lstm_size, input_shape=input_data[0].shape, return_sequences=True))
    model.add(keras.layers.LSTM(lstm_size))
    model.add(keras.layers.Dense(len(mapping), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    # Train the network
    cp_callback = keras.callbacks.ModelCheckpoint(filepath=('checkpoints/' + save_file.split('.')[0] + '.ckpt'), 
                                                  save_weights_only=True, save_best_only=True, verbose=1)
    model.fit(input_data, output_data, epochs=epochs, batch_size=batch_size,
              validation_split=validation_split, callbacks=[cp_callback])
    _, accuracy = model.evaluate(input_data, output_data)
    print('Accuracy: %.2f%%' % (accuracy * 100))
    model.save(model_file)


def generate_text(save_file, num_lines, max_limit, words=False):
    # Read save file
    with open(save_file, 'r') as f:
        lines = [line.strip() for line in f]
        input_length = int(lines[0])
        model_file = lines[1]
        mapping = list('\n'.join(lines[2:]).split(';#;'))
    one_hot = np.identity(len(mapping))
    # Prepare model
    model = keras.models.load_model(model_file)
    seq = [one_hot[random.randint(0, len(mapping)-1)] for _ in range(input_length-1)]
    seq += [one_hot[mapping.index('\n')]]
    # Generate text
    result = list()
    current_line = 1
    while current_line <= num_lines and len(seq) <= input_length + max_limit:
        rankings = model.predict(np.array([seq[-input_length:]]))[0].tolist()
        index = rankings.index(max(rankings))
        seq.append(one_hot[index])
        result.append(mapping[index])
        if result[-1] == '\n':
            print('Line', current_line, 'out of', num_lines, 'complete.')
            current_line += 1
    print('\n--------------------------------------------\n')
    text = (' ' if words else '').join(result)
    print(text)
    with open(os.path.join(os.getcwd(), 'most_recent_text.txt'), 'w+') as f:
        f.write(text)


if __name__ == '__main__':
    # train_data('Christmas Songs/', 'Christmas1.txt', input_length=100, lstm_size=700, epochs=5, batch_size=50,
    #            validation_split=0.1, valid_punctuation=['\n', ' ', ',', '(', ')', '-'], words=False)
    generate_text('Christmas1.txt', 24, 50)
