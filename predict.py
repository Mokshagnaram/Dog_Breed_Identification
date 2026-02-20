import os
from functools import lru_cache

import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.preprocessing.image import img_to_array

MODEL_PATH = "model/dog_breed_model.h5"
DATASET_TRAIN_DIR = "dataset/train"
IMAGE_SIZE = (160, 160)


@lru_cache(maxsize=1)
def load_model(model_path: str = MODEL_PATH):
    """Load and cache the trained Keras model."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. Train the model first."
        )
    return keras_load_model(model_path)


@lru_cache(maxsize=1)
def load_class_names(train_dir: str = DATASET_TRAIN_DIR):
    """Load class names from dataset/train directory in sorted order."""
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"Training dataset directory not found at '{train_dir}'."
        )

    class_names = sorted(
        [
            folder
            for folder in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, folder))
        ]
    )

    if not class_names:
        raise ValueError("No class folders found in dataset/train.")

    return class_names


def preprocess_image(image_path: str) -> np.ndarray:
    """Load image from path, resize, normalize, and add batch dimension."""
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        image = Image.open(image_path).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Invalid image file. Please upload a valid image.") from exc

    image = image.resize(IMAGE_SIZE)
    image_array = img_to_array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def predict_breed(image_path: str):
    """
    Predict dog breed for an image.

    Returns:
        tuple[str, float]: (predicted_breed, confidence_percent)
    """
    model = load_model()
    class_names = load_class_names()

    processed_image = preprocess_image(image_path)
    prediction = model.predict(processed_image, verbose=0)[0]

    predicted_index = int(np.argmax(prediction))
    confidence = float(prediction[predicted_index] * 100)

    if predicted_index >= len(class_names):
        raise ValueError("Model output classes do not match dataset class folders.")

    predicted_breed = class_names[predicted_index]
    return predicted_breed, round(confidence, 2)
