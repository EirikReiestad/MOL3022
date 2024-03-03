from PIL import Image
import tensorflow.keras as keras
from predictor import PSSPredictor


def create_model_plot(model, filename="model.png", show=True):
    """
    Create a plot of the model
    """
    keras.utils.plot_model(
        model,
        to_file=filename,
        show_shapes=True,
    )

    if show:
        img = Image.open(filename)
        img.show()


if __name__ == "__main__":
    WINDOW_SIZE = 17  # Doesn't matter for this example
    predictor = PSSPredictor(WINDOW_SIZE)
    print("Creating model plot...")
    create_model_plot(predictor.model)
