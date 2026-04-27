# 🧠 Digit Recognition App (Python + GUI)

A desktop application that uses Machine Learning to recognize digits from uploaded images.

Built with Python, Scikit-learn, Tkinter, Pillow, NumPy, and Joblib.

---

## 🚀 Features

- 📂 Upload images (PNG / JPG / JPEG)
- 🖼️ Live image preview in GUI
- 🔍 Automatic digit prediction
- 🧠 SVM-based machine learning model
- 💾 Model saving & loading using Joblib
- 🖨️ Supports simple printed and handwritten-style digits
- ⚡ No internet required after setup

---

## 🛠️ Tech Stack

- Python
- Scikit-learn
- Tkinter
- Pillow
- NumPy
- Joblib

---

## 📸 How It Works

The model is trained using Scikit-learn’s built-in digits dataset plus synthetic printed digit images.

Uploaded images are:

- Converted to grayscale
- Cropped and padded
- Resized to 28×28 pixels
- Normalized

Then the trained SVM model predicts the digit and displays the result in the GUI.

---

## ▶️ Run the Project

```bash
pip install scikit-learn pillow joblib numpy
python main.py
