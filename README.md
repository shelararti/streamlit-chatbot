# streamlit-chatbot
🤖 A simple chatbot built with PyTorch and Streamlit, featuring fuzzy matching and a web-based UI for interactive conversations.

---

## 🧠 Features

- Natural language input
- Fuzzy matching using RapidFuzz
- Trained on custom intents
- Interactive Streamlit web interface

---

## 🚀 How to Run

1. Clone the repo:

git clone https://github.com/your-username/chatbot-app.git
cd chatbot-app
Install dependencies:


pip install -r requirements.txt
Run the chatbot UI:


streamlit run app.py

---

## 📁 Files
train.py – Trains the chatbot model

chat.py – Handles chatbot responses

app.py – Streamlit interface

intents.json – Training data

data.pth – Trained model

---

### ✅ **Step 4: Initialize Git and Push to GitHub**

1. Open a terminal in your project folder and run:

git init
git add .
git commit -m "Initial commit - chatbot app"
Create a GitHub repo (e.g., chatbot-app) at github.com.

2. Link your local repo to GitHub:

git remote add origin https://github.com/your-username/chatbot-app.git
git branch -M main
git push -u origin main
