import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed
from tensorflow.keras import layers
from tensorflow import keras

tf.keras.utils.set_random_seed(812)

pd.set_option("future.no_silent_downcasting", True)

# @tf.keras.utils.register_keras_serializable(package='Custom', name='PSSPredictor')


class PSSPredictor:
    def __init__(
        self,
        window_size,
        protein_letters="ACDEFGHIKLMNPQRSTVWXY",
        secondary_letters="ceh",
    ):
        self.window_size = window_size
        self.protein_letters = protein_letters
        self.secondary_letters = secondary_letters
        self.model = self.create_model()

    def create_model(self):
        model = Sequential(
            [
                LSTM(
                    32,
                    input_shape=(self.window_size, len(self.protein_letters)),
                    return_sequences=True,
                ),
                layers.BatchNormalization(),
                layers.Dropout(0.4),
                Dense(96, activation="tanh"),
                TimeDistributed(
                    Dense(len(self.secondary_letters), activation="softmax")
                ),
            ]
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.051000000000000004),
            loss="categorical_crossentropy",
            metrics=["accuracy", "mae", q3_score],
        )

        return model

    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
    ):
        if not os.path.exists("model.h5"):
            history = self.model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
            )
            self.model.save("model.h5")
        else:
            self.model = self.load_model("model.h5")
            history = None
        loss, accuracy, mae, q3 = self.model.evaluate(X_val, y_val)
        print(f"Validation Loss: {loss}, Accuracy: {accuracy}, MAE: {mae}, Q3: {q3}")
        return history, loss, accuracy, mae, q3

    def predict(self, X):
        return self.model.predict(X)

    @staticmethod
    def load_model(filepath="model.h5"):
        # Load the model from the file
        loaded_model = tf.keras.models.load_model(
            filepath, custom_objects={"q3_score": q3_score}
        )
        print("Model loaded successfully.")
        return loaded_model


@tf.keras.utils.register_keras_serializable(package="Custom", name="q3_score")
def q3_score(target, prediction):
    target_labels = tf.argmax(target, axis=-1)
    prediction_labels = tf.argmax(prediction, axis=-1)
    q3 = tf.reduce_mean(
        tf.cast(tf.equal(target_labels, prediction_labels), dtype=tf.float32)
    )
    return q3
