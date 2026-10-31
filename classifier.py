from pathlib import Path
import numpy as np
from PIL import Image

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMAGE_SIZE = (224, 224)

def load_classifier(model_path):
    """Load a trained Keras model. Returns None when no model exists."""
    path = Path(model_path)
    if not path.exists():
        return None
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(path)
    except Exception as exc:
        raise RuntimeError(
            "Could not load the model. Check that TensorFlow is installed and "
            "the model file is valid."
        ) from exc

def predict_image(model, image):
    """Return (class_name, confidence) for one PIL image."""
    import tensorflow as tf
    image = image.convert("RGB").resize(IMAGE_SIZE)
    arr = np.asarray(image, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    probs = model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx])
