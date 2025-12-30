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

### 1️⃣ Clone the Repository
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
3️⃣ Run the App
Copy code
Bash
streamlit run app.py
The app will open in your browser. Upload a lunar surface image to see predictions.
🎨 Video Demo

[Watch Demo Video](https://drive.google.com/file/d/1e8u-nex5xmCsTRUhfGq5xlbkMHbNBsKw/view?usp=sharing)


🧩 Project Structure
Copy code

Lunar_ML_Project/
├─ app.py                  # Streamlit application
├─ lunar_cnn_model.keras   # Pretrained CNN model (in GitHub release)
├─ requirements.txt        # Python dependencies
├─ README.md               # This file
├─ Dockerfile              # Optional container setup
└─ .gitattributes
⚠️ Notes
The CNN model file (lunar_cnn_model.keras) is large, stored in GitHub Releases. The app downloads it automatically on first run.
Ensure you run streamlit run app.py in the same folder as app.py and the model.
Tested locally; cloud deployment may require additional configuration (e.g., Hugging Face Spaces or Google Colab).
💡 Future Improvements
Deploy on free cloud platforms (Hugging Face Spaces, Streamlit Cloud) with automatic model download
Enhance UI/UX with sliders or image history
Extend classification to include lunar crater depth estimation or other surface features
Incorporate real-time lunar imagery from NASA datasets
🧑‍💻 Author
Lekshmi 
📄 License
MIT License – See LICENSE for details.
