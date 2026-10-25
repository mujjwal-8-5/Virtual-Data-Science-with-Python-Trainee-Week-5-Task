"""
Week 5: Deep Learning Application in Data Science
Fashion-MNIST image classification with TensorFlow/Keras.

Run:
    pip install -r requirements.txt
    python src/train.py

The script downloads Fashion-MNIST through Keras on first run (internet required),
trains a CNN, prints evaluation metrics, and writes results/predictions.csv,
results/metrics.json, and figures/training_curves.png.
"""
from pathlib import Path
import json
import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

SEED = 42
tf.keras.utils.set_random_seed(SEED)

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def main():
    # 1. Load the public Fashion-MNIST dataset (60,000 train / 10,000 test images).
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # 2. Normalize pixels from [0, 255] to [0, 1] and add the channel dimension.
    x_train = (x_train.astype("float32") / 255.0)[..., np.newaxis]
    x_test = (x_test.astype("float32") / 255.0)[..., np.newaxis]

    # 3. Keep a validation split separate from the final held-out test set.
    # Keras takes the last 10% of the training arrays as validation data.
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.30),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(RESULTS / "best_model.keras"),
            monitor="val_loss", save_best_only=True
        )
    ]

    history = model.fit(
        x_train, y_train,
        epochs=15,
        batch_size=128,
        validation_split=0.10,
        callbacks=callbacks,
        verbose=2
    )

    # 4. Evaluate once on the held-out test set.
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    probabilities = model.predict(x_test, batch_size=256, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    report_dict = classification_report(
        y_test, predictions, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_test, predictions)
    metrics = {
        "dataset": "Fashion-MNIST",
        "seed": SEED,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "epochs_trained": len(history.history["loss"]),
        "class_names": CLASS_NAMES,
        "classification_report": report_dict,
        "confusion_matrix": matrix.tolist()
    }
    with open(RESULTS / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(RESULTS / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "true_label", "true_class", "predicted_label",
                         "predicted_class", "confidence", "correct"])
        for i, (true_label, pred_label) in enumerate(zip(y_test, predictions)):
            writer.writerow([
                i, int(true_label), CLASS_NAMES[int(true_label)], int(pred_label),
                CLASS_NAMES[int(pred_label)], float(np.max(probabilities[i])),
                bool(true_label == pred_label)
            ])

    # 5. Plot training curves.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["loss"], label="Training loss")
    axes[0].plot(history.history["val_loss"], label="Validation loss")
    axes[0].set_title("Loss by epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()
    axes[1].plot(history.history["accuracy"], label="Training accuracy")
    axes[1].plot(history.history["val_accuracy"], label="Validation accuracy")
    axes[1].set_title("Accuracy by epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "training_curves.png", dpi=160)
    plt.close(fig)

    disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(10, 9))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False, values_format="d")
    ax.set_title("Fashion-MNIST test confusion matrix")
    fig.tight_layout()
    fig.savefig(FIGURES / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    print("\n=== FINAL TEST RESULTS ===")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4%}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, target_names=CLASS_NAMES, zero_division=0))
    print(f"\nSaved metrics to: {RESULTS / 'metrics.json'}")
    print(f"Saved predictions to: {RESULTS / 'predictions.csv'}")
    print(f"Saved plots to: {FIGURES}")

if __name__ == "__main__":
    main()
