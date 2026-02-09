# 🧠 NeuroGuard – Mental State AI Assistant

> *"In 2080, your mind is the battlefield. NeuroGuard is your last line of defence."*

---

## 🎥 Demo

👉 https://github.com/user-attachments/assets/d4a73a8d-0e5e-4a19-b743-6662fb6ba338



## 💡 Project Name: NeuroGuard

---

## 🏁 Hackathon Context

**NeuroGuard** was developed during **Great UniHack 2025 (Manchester)** — a university hackathon focused on **future challenges and speculative problem-solving**.

The challenge encouraged teams to:
- identify a **realistic problem of the future**, and
- design a **theoretical or experimental system** that could address it.

Our team chose to focus on a challenge we believe will define the coming decades:

> **Mental health in an era of extreme technological saturation.**

---

## 🛡️ The Solution

**NeuroGuard** is a lightweight AI-powered web application that estimates a user’s mental state based on everyday lifestyle signals and provides actionable recommendations to prevent overload and burnout.

The system:
- collects structured lifestyle inputs,
- estimates a **mental state score (0–100)** using a machine learning model,
- generates **context-aware suggestions** to support cognitive balance.

The application is designed to run **locally**, with optional AI-enhanced recommendations.

---

## 🧠 How It Works (High Level)

1. The user enters lifestyle information (sleep, screen time, activity, daylight, etc.).
2. A trained **RandomForestRegressor** estimates the mental state score.
3. Based on the prediction:
   - offline fallback recommendations are generated, or
   - optional **Gemini API** suggestions are provided.
4. The result is displayed instantly in a clean, cyberpunk-inspired UI.
5. User history is stored locally for future analysis.

---

## 🔑 Core Features

- **🧠 Mental State Estimation**  
  Machine learning–based prediction using a `RandomForestRegressor`.

- **🧬 AI-Driven Recommendations (Optional)**  
  Context-aware mental wellness tips via Google Gemini API, with full offline fallback.

- **💻 Interactive Web Interface**  
  Modern, responsive frontend built with HTML, CSS, and JavaScript.

- **💾 Local Persistence**  
  User inputs and summaries stored in a local SQLite database.

- **🔐 Privacy by Design**  
  Runs locally by default. External AI integration is optional and transparent.

- **✅ Robust Validation & UX**  
  Input validation, range checks, loading states, and session-safe navigation.

---

#### 🧰 Tech Stack

**Languages**
- Python
- HTML, CSS, JavaScript

**Backend**
- Flask
- SQLite
- python-dotenv

**Machine Learning**
- scikit-learn
- pandas
- RandomForestRegressor

**AI Integration (Optional)**
- Google Gemini API
  
### 🛠️ Installation Instructions:
To get started, make sure you have **Python 3.9+** installed. Then clone the repository and install all required dependencies using the following command:

```bash
pip install -r requirements.txt
```

After that, create a `.env` file in the root directory of the project (see .env.example) and add your **Gemini API Key** in the following format:

```env
FLASK_SECRET_KEY=your-secret-key
GEMINI_API_KEY=your-gemini-api-key   # optional
```

Then launch the application with:

```bash
python main.py
```

---

#### 🚀 Usage:
1. Visit: `http://localhost:4000`  
2. Submit lifestyle info → Receive mental state score  
3. Trigger Gemini-powered suggestions  
4. Retrain the model using CLI: `python main.py`

---


### 🎯 Why It Matters

NeuroGuard is not a medical device.
It is a **conceptual exploration** of digital well-being in an over-optimised future.

As technology increasingly shapes cognition, projects like NeuroGuard highlight the importance of:
- mental resilience,
- human-centered system design,
- ethical AI integration.

---

> *"They enhanced our bodies. We built NeuroGuard to protect what’s left of the mind."*


## 👥 Team

- Dmytro Dudarenko
- Kamron Khusainov
- Thomas Palmer


#### 📌 Notes

- This project was developed as a team-based exploratory prototype.
- The ML model is trained on structured sample data for demonstration purposes.
- Designed for educational, experimental, and portfolio use.
