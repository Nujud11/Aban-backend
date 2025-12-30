# Aban — ADOS Score Prediction Backend

Aban is a graduation project that focuses on predicting **ADOS (Autism Diagnostic Observation Schedule) scores** for children with Autism Spectrum Disorder (ASD) using deep learning models trained on behavioral time-series data.

This repository contains the **backend system**, which loads a trained deep learning model and provides prediction results through a Flask-based web API.

---

## 🧠 Project Overview

Early and accurate assessment of autism severity is essential for effective intervention. Traditional ADOS assessment relies heavily on manual observation and expert judgment, which can be time-consuming and subjective.

Aban aims to support specialists by providing an **automated, data-driven prediction system** based on:
- Eye-gaze behavior
- Upper-body movement patterns
- Sequential 3D behavioral data

The backend serves as the core inference engine of the system.

---

## ⚙️ Backend Responsibilities

- Load trained deep learning models (Bi-LSTM / RNN)
- Receive processed behavioral input data
- Perform ADOS score prediction
- Return results to the web interface
- Serve as an API endpoint for the frontend

---

## 🧪 Model Details

- Architecture: **Bi-directional LSTM (Bi-LSTM)** and RNN
- Input: Sequential eye-gaze and skeletal movement features
- Output: Predicted ADOS score / severity level
- Frameworks: TensorFlow / Keras

The trained model is stored as an `.h5` file and loaded at runtime.

---

## 🌐 Live Services

- **Backend API (Render):**  
  https://aban-backend.onrender.com

- **Frontend (GitHub Pages):**  
  https://nujud11.github.io/aban-website/

> ⚠️ Note: The backend is deployed on a free-tier hosting service and may take a few seconds to wake up if idle.

---

## 🧩 Related Repositories

- Frontend Repository:  
  https://github.com/Nujud11/aban-website

---

## 🛠 Tech Stack

- Python
- Flask
- TensorFlow / Keras
- NumPy
- HTML / CSS (for integrated views)

---

## ▶️ Run Locally

```bash
git clone https://github.com/Nujud11/Aban-backend.git
cd Aban-backend

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python app.py
