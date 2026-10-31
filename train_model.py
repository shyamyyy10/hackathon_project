"""Train a six-class waste image classifier with transfer learning.

Expected dataset layout:
data/garbage_classification/
  cardboard/ image1.jpg ...
  glass/ ...
  metal/ ...
  paper/ ...
  plastic/ ...
  trash/ ...

Install dependencies first, then run:
python train_model.py --data_dir data/garbage_classification --epochs 8
"""
import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/garbage_classification")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--output", default="models/waste_classifier.keras")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(
            f"Dataset folder not found: {data_dir}\\n"
            "Download a compatible dataset and arrange it into the six class folders. "
            "See README.md."
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="training", seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, class_names=CLASSES
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="validation", seed=SEED,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE, class_names=CLASSES
    )
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)

    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
    ])
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3), include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    # Fine-tune a small portion of the pretrained backbone for a few epochs.
    base.trainable = True
    for layer in base.layers[:-25]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(train_ds, validation_data=val_ds, epochs=max(1, min(3, args.epochs // 3 or 1)))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    print(f"Saved model to {out}")
    print("Class order:", CLASSES)

if __name__ == "__main__":
    main()
