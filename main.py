from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import joblib
import numpy as np
from PIL import Image, ImageOps, ImageTk, ImageDraw, ImageFont

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


MODEL_FILE = "digit_model.joblib"
IMG_SIZE = 28


def preprocess_image(img):
    img = img.convert("L")

    arr = np.array(img)

    # If background is white, invert image
    if arr.mean() > 127:
        img = ImageOps.invert(img)

    arr = np.array(img)

    # Find digit area
    coords = np.argwhere(arr > 30)

    if coords.size > 0:
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        img = img.crop((x0, y0, x1 + 1, y1 + 1))

    # Make square canvas
    w, h = img.size
    size = max(w, h)
    canvas = Image.new("L", (size, size), 0)

    x = (size - w) // 2
    y = (size - h) // 2
    canvas.paste(img, (x, y))

    # Add padding
    canvas = ImageOps.expand(canvas, border=8, fill=0)

    # Resize to 28x28
    canvas = canvas.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)

    arr = np.array(canvas).astype("float32") / 255.0

    return arr.flatten()


def load_windows_fonts():
    font_paths = []

    fonts_dir = Path("C:/Windows/Fonts")

    if fonts_dir.exists():
        for ext in ["*.ttf", "*.otf"]:
            font_paths.extend(fonts_dir.glob(ext))

    return font_paths[:80]


def create_synthetic_printed_digits():
    X = []
    y = []

    font_paths = load_windows_fonts()

    for digit in range(10):
        for font_path in font_paths:
            for font_size in [26, 32, 38, 44, 52]:
                for angle in [-12, -6, 0, 6, 12]:
                    try:
                        font = ImageFont.truetype(str(font_path), font_size)
                    except:
                        continue

                    img = Image.new("L", (80, 80), 255)
                    draw = ImageDraw.Draw(img)

                    text = str(digit)
                    bbox = draw.textbbox((0, 0), text, font=font)

                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]

                    x = (80 - tw) // 2
                    y_pos = (80 - th) // 2

                    draw.text((x, y_pos), text, fill=0, font=font)

                    img = img.rotate(angle, fillcolor=255)

                    features = preprocess_image(img)
                    X.append(features)
                    y.append(digit)

    return np.array(X), np.array(y)


def create_handwritten_digits():
    digits = load_digits()

    X = []
    y = []

    for image, label in zip(digits.images, digits.target):
        img_arr = (image / 16.0 * 255).astype("uint8")
        img = Image.fromarray(img_arr, mode="L")
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)

        arr = np.array(img).astype("float32") / 255.0
        X.append(arr.flatten())
        y.append(label)

    return np.array(X), np.array(y)


def train_model():
    print("Creating training data...")

    X_hand, y_hand = create_handwritten_digits()
    X_print, y_print = create_synthetic_printed_digits()

    X = np.vstack([X_hand, X_print])
    y = np.concatenate([y_hand, y_print])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True
    )

    print("Training model...")
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print("Accuracy:", acc)

    joblib.dump(model, MODEL_FILE)
    print("Model saved.")

    return model


def load_or_train_model():
    if Path(MODEL_FILE).exists():
        print("Loading saved model...")
        model = joblib.load(MODEL_FILE)

        if hasattr(model, "n_features_in_") and model.n_features_in_ != IMG_SIZE * IMG_SIZE:
            print("Old model found. Retraining...")
            Path(MODEL_FILE).unlink()
            return train_model()

        return model

    return train_model()

def predict_digit(image_path):
    model = load_or_train_model()

    img = Image.open(image_path)
    features = preprocess_image(img)

    prediction = model.predict([features])[0]

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([features])[0]
        confidence = probs.max() * 100
    else:
        confidence = 0

    return prediction, confidence


def upload_image():
    file_path = filedialog.askopenfilename(
        title="Select digit image",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("JPG files", "*.jpg *.jpeg")
        ]
    )

    if not file_path:
        return

    try:
        img = Image.open(file_path)
        img.thumbnail((250, 250))

        preview = ImageTk.PhotoImage(img)
        image_label.config(image=preview)
        image_label.image = preview

        result, confidence = predict_digit(file_path)

        result_label.config(
            text=f"Predicted Digit: {result}\nConfidence: {confidence:.2f}%"
        )

    except Exception as e:
        messagebox.showerror("Error", str(e))


root = tk.Tk()
root.title("Digit Recognition")
root.geometry("460x580")
root.resizable(False, False)

title_label = tk.Label(
    root,
    text="Digit Recognition",
    font=("Arial", 22, "bold")
)
title_label.pack(pady=15)

upload_button = tk.Button(
    root,
    text="Upload Image",
    font=("Arial", 14),
    command=upload_image
)
upload_button.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=15)

result_label = tk.Label(
    root,
    text="Predicted Digit: -",
    font=("Arial", 20, "bold"),
    fg="blue"
)
result_label.pack(pady=20)

note_label = tk.Label(
    root,
    text="Supports PNG, JPG, JPEG | Printed + handwritten digits",
    font=("Arial", 10)
)
note_label.pack(pady=5)

root.mainloop()
