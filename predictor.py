import pandas as pd
import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed, Embedding, Bidirectional
from tensorflow.keras import layers

tensorflow.keras.utils.set_random_seed(812)

pd.set_option('future.no_silent_downcasting', True)

class PSSPredictor():
    def __init__(self, window_size, protein_letters='ACDEFGHIKLMNPQRSTVWXY', secondary_letters='ceh'):
        self.window_size = window_size
        self.protein_letters = protein_letters
        self.secondary_letters = secondary_letters
        self.model = self.create_model()

    def create_model(self):
        drop_out = 0.3

        model = Sequential([
            LSTM(128, input_shape=(self.window_size, len(self.protein_letters)), return_sequences=True),
            layers.BatchNormalization(),
            layers.Dropout(drop_out),
            Dense(64, activation='tanh'),
            layers.BatchNormalization(),
            layers.Dropout(drop_out),
            Dense(128, activation='linear'),
            layers.BatchNormalization(),
            layers.Dropout(drop_out),
            Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(drop_out),
            Dense(32, activation='relu'),
            TimeDistributed(Dense(len(self.secondary_letters), activation='softmax')) # Apply Dense layer to each time step
        ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'mae'])

        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, validation_split=0.2):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        loss, accuracy, mae = self.model.evaluate(X_val, y_val)
        print(f'Validation Loss: {loss}, Accuracy: {accuracy}, MAE: {mae}')
    
    def predict(self, X):
        return self.model.predict(X)