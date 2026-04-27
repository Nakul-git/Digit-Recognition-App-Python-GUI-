# 🧠 Digit Recognition App (Python + GUI)

A simple desktop application that uses Machine Learning to recognize handwritten digits from uploaded images.

Built with Python, Scikit-learn, and Tkinter, this project demonstrates the complete ML workflow — from training a model to deploying it in a GUI.

---

## 🚀 Features

- 📂 Upload image (PNG / JPG / JPEG)
- 🖼️ Live image preview in GUI
- 🔍 Automatic digit prediction
- 🧠 Trained ML model (Neural Network / SVM)
- 💾 Model saving & loading (joblib)
- ⚡ No internet required (uses built-in dataset)

---

## 🛠️ Tech Stack

- Python  
- Scikit-learn  
- Tkinter (GUI)  
- Pillow (Image processing)  
- Joblib (Model persistence)  

---

## 📸 How It Works

1. Model is trained using the **Digits dataset** (`sklearn.datasets`)
2. Uploaded image is:
   - Converted to grayscale  
   - Cropped & padded  
   - Resized to 8×8  
   - Normalized  
3. Model predicts the digit  
4. Result is displayed in GUI  

---

## ▶️ Run the Project

```bash
pip install scikit-learn pillow joblib
python main.py
