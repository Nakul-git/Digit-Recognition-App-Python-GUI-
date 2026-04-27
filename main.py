from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import joblib
from PIL import Image, ImageOps, ImageTk
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


MODEL_FILE = "digit_model.joblib"


def train_model():
    digits = load_digits()

    X = digits.data / 16.0
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = SVC(
    kernel="rbf",
    gamma=0.001,
    C=10
    )

    print("Training model...")
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_FILE)
    print("Model saved.")

    return model


def load_or_train_model():
    if Path(MODEL_FILE).exists():
        print("Loading saved model...")
        return joblib.load(MODEL_FILE)

    return train_model()


def prepare_image(image_path):
    img = Image.open(image_path).convert("L")

    pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)

    if avg > 127:
        img = ImageOps.invert(img)

    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)

    img = ImageOps.expand(img, border=20, fill=0)
    img = img.resize((8, 8), Image.Resampling.LANCZOS)
    img = img.point(lambda p: 255 if p > 40 else 0)

    pixels = list(img.getdata())
    pixels = [p / 255.0 for p in pixels]

    return [pixels]


def predict_digit(image_path):
    model = load_or_train_model()
    image_data = prepare_image(image_path)
    prediction = model.predict(image_data)
    return prediction[0]


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

        result = predict_digit(file_path)
        result_label.config(text=f"Predicted Digit: {result}")

    except Exception as e:
        messagebox.showerror("Error", str(e))


root = tk.Tk()
root.title("Digit Recognition")
root.geometry("430x520")
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
    text="Supports PNG, JPG, JPEG",
    font=("Arial", 10)
)
note_label.pack(pady=5)

root.mainloop()