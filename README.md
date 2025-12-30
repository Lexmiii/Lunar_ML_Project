# 🌕 Lunar Surface Classification

## Overview
A **CNN-based Machine Learning project** that classifies lunar surface images into **Smooth** or **Cratered** surfaces and estimates **Landing Risk**. Built with **TensorFlow** and **Streamlit**, this project demonstrates practical image classification with a clean, interactive web interface.

> **Motivation:** Understanding lunar surfaces is crucial for space missions. Automating surface detection can help scientists and engineers evaluate landing zones efficiently.

---

## 🛠 Components Used
- **Python 3.13+**  
- **TensorFlow** – Convolutional Neural Network model  
- **Streamlit** – Web app for interactive visualization  
- **NumPy & Pillow** – Image preprocessing  
- **Pretrained Model** – `lunar_cnn_model.keras`  

---

## ✨ Features
- Upload lunar surface images (JPG / PNG)  
- Predict **Surface Type**: Smooth / Crater  
- Display **Landing Risk**: Low Risk / High Risk  
- Show **Confidence Score** for each prediction  
- Responsive **Streamlit UI** with modern design

---

## 📂 Project Setup
 1️⃣ Clone the Repository
Open your terminal (or VS Code terminal) and run:

```bash
git clone https://github.com/Lexmiii/Lunar_ML_Project.git
cd Lunar_ML_Project

2️⃣ Install Dependencies
Make sure Python 3.13+ is installed:
Copy code
Bash
pip install -r requirements.txt
This will install Streamlit, TensorFlow, NumPy, Pillow, and other required packages.
