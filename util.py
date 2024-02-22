import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

PROTEIN_LETTERS = 'ACDEFGHIKLMNPQRSTVWXY'
SECONDARY_LETTERS = 'ceh'


def read_protein_file(file_path):
    '''
    Read protein file
    '''
    with open(file_path, "r") as file:
        # Read the entire contents of the file
        seq = ""
        str = ""

        sequences = []
        strings = []
        count = 0
        for line in file:
            if count < 8:
                pass
            count += 1

            line = line.strip()
            if line == "<>":
                seq = ""
                str = ""
            elif line == "<end>" or line == "end":
                sequences.append(seq)
                strings.append(str)
            else:
                letters = line.split(" ")
                if len(letters) == 2:
                    seq += letters[0]
                    str += letters[1] if letters[1] != "_" else "c"
        return sequences, strings


def split_based_on_windows(data_seq, data_str=None, window_size=17):
    '''
    Split sequences into windows of size window_size
    '''
    all_sequences = []
    all_strings = []
    for i in range(len(data_seq)):
        sequences = [data_seq[i][j:j+window_size]
                     for j in range(0, len(data_seq[i]), window_size)]
        all_sequences += sequences

        if data_str is not None:
            strings = [data_str[i][j:j+window_size]
                       for j in range(0, len(data_str[i]), window_size)]
            all_strings += strings

    if len(all_strings) == 0:
        df = pd.DataFrame({"sequence": all_sequences})
    else:
        df = pd.DataFrame({"sequence": all_sequences, "string": all_strings})
    return df


def ohe_for_nn(sequences, strings=None):
    '''
    One hot encoding for input to neural network
    '''
    X_ohe = [[PROTEIN_LETTERS.index(letter)
              for letter in seq] for seq in sequences]
    max_length = max(len(seq) for seq in X_ohe)
    X_padded = pad_sequences(X_ohe, maxlen=max_length, padding='post')
    X = np.zeros((len(X_padded), max_length, len(PROTEIN_LETTERS)))

    if strings is not None:
        y_ohe = [[SECONDARY_LETTERS.index(letter)
                  for letter in string] for string in strings]
        y_padded = pad_sequences(y_ohe, maxlen=max_length, padding='post')
        y = np.zeros((len(y_padded), max_length, len(SECONDARY_LETTERS)))

    for i in range(len(X_padded)):
        for j, aa_index in enumerate(X_padded[i]):
            X[i, j, aa_index] = 1
        if strings is not None:
            for j, structure_index in enumerate(y_padded[i]):
                y[i, j, structure_index] = 1

    if strings is not None:
        return X, y
    else:
        return X


def convert_pred_to_str(predictions):
    '''
    Convert NN prediction back to string
    '''
    inv_structure_map = {0: 'c', 1: 'e', 2: 'h'}
    y_pred_classes = np.argmax(predictions, axis=-1)
    protein_array = np.vectorize(inv_structure_map.get)(y_pred_classes)

    array_strings = ["".join(map(str, row)) for row in protein_array]
    return array_strings


def create_plot(sequence, structure):
    '''
    Create plot to visualize the secondary structure
    '''
    _, ax = plt.subplots(figsize=(10, 2))

    structure_letters = ['c', 'h', 'e']
    ax.set_yticks(range(len(structure_letters)))
    ax.set_yticklabels(structure_letters)

    ax.set_xticks(range(len(sequence)))
    ax.set_xticklabels(sequence)

    for i in range(len(sequence)):
        structure_type = structure[i]
        if structure_type == 'c':
            color = 'skyblue'
        elif structure_type == 'h':
            color = 'salmon'
        else:
            color = 'lightgreen'
        rect = Rectangle((i - 0.5, structure_letters.index(structure_type) -
                         0.5), 1, 1, facecolor=color, edgecolor='black')
        ax.add_patch(rect)

    ax.set_aspect('equal')

    plt.xticks()
    plt.grid(True, which='both', linestyle='--',
             linewidth=0.5)  # Add gridlines
    plt.tight_layout()
    plt.savefig('./images/prediction.png')
