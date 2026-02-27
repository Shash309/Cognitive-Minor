# 🤖 CAPE — Career Path Explorer

<p align="center">
  <img src="./assets/hero.gif" alt="CAPE Hero" width="760" style="border-radius:12px;box-shadow:0 8px 30px rgba(11,22,39,0.12)"/>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue"/>
  <img alt="React" src="https://img.shields.io/badge/react-19-blue"/>
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green"/>
  <img alt="Status" src="https://img.shields.io/badge/status-Active-brightgreen"/>
</p>

**CAPE (Career Path Explorer)** is an advanced, AI-driven career recommendation engine and path visualizer. It seamlessly maps student profiles—comprising skills, education, and psychological traits—to ranked career suggestions and generates interactive educational roadmaps to guide them toward their goals.

---

## ✨ Core Features & Modules

CAPE provides an end-to-end career guidance ecosystem equipped with cutting-edge ML pipelines and an engaging modern interface.

### 🧠 ML-Powered Recommendation Engine
At the heart of CAPE lies a sophisticated prediction engine that fuses multiple models:
- **Natural Language Processing (NLP)**: Uses `SentenceTransformer (all-MiniLM-L6-v2)` to generate embeddings from text-based quiz responses.
- **Classification Model**: Integrates TF-IDF features with embeddings to classify responses using a trained Random Forest model (`career_1200_model.pkl`).
- **Fusion Scoring**: Merges ML-derived text analysis with psychometric models to accurately rank and align matching careers (Data Scientist, Software Engineer, Manager, Designer, etc.).

### 📊 Psychological Assessment
CAPE performs dynamic psychometric evaluations based on the widely recognized **Big Five Personality Traits** (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) along with custom cognitive vectors:
- **Trait Vectors**: Evaluates Decision-Making Style, Stress Tolerance, Risk Tolerance, and Analytical vs. Intuitive thinking.
- **Skill Gap Analysis**: Compares a user’s current traits against the baseline requirements of selected careers, identifying areas (e.g., Extraversion, Leadership) where the user can focus their growth.
- **Radar Charts**: Beautifully visualizes dominant psychological profiles in the dashboard.

### 🗺️ Visualizer 2.0: Interactive Career Journey
Navigate through careers natively using a mind-map styled interface powered by **React Flow** and **ELK.js**.
- **Directional Path Tracing**: Click on any educational or career stage to highlight the entire journey from root to destination.
- **Animated Dotted Connections**: Visual guidance showing the flow of progression.
- **Path Glow**: Selected nodes emit a high-intensity glow for clear focus.

### 🛤️ Roadmap Timeline & Skill Builder
Once a career path is chosen, CAPE dynamically generates a step-by-step **Roadmap Timeline** outlining specific milestones (e.g., getting a degree, building a portfolio, completing an internship). 
Coupled with the **Skill Builder** module, users receive actionable, targeted recommendations on what hard and soft skills are required to succeed at every stage.

### 🎓 College Explorer
A dedicated interface integrating a customized database (`colleges.csv`) to help users search and filter relevant colleges that offer the degrees necessary for their chosen career paths.

### 💎 Stunning Modern UI
- **3D Interactive Backgrounds**: Uses **Vanta.js** and **Three.js** to render beautiful, immersive fluid 3D backgrounds.
- **Fluid Animations**: State transitions and component mounting are naturally animated using **Framer Motion**.
- **Bento-Box Inspiration**: Features an accessible, card-based layout providing a premium User Experience tailored for responsiveness.

---

## 🛠️ Technology Stack

**Frontend (Client-Side)**
- **Framework**: React 19 + Vite
- **Styling & UI**: Modern Vanilla CSS, Framer Motion (Animations), react-icons
- **Visualization**: React Flow, ELK.js (Layout Algorithms), Three.js, Vanta.js
- **Routing**: React Router DOM
- **Internationalization**: i18next

**Backend (Server & ML)**
- **Framework**: Python 3.10+, Flask, Flask-CORS
- **Machine Learning**: Scikit-Learn, Sentence-Transformers, PyTorch
- **Data Manipulation**: Pandas, NumPy
- **Storage**: Lightweight JSON databases for rapid stateless architecture.

---

## 📁 Project Architecture

The repository is modularly split into a Python API layer and a React UI layer:

```
.vscode/                     - Editor settings (formatting, extensions)
backend/                     - Python API (Flask) & ML Pipeline
  ├── data/                  - DB storage, user responses, psych_profiles, and collages.csv
  ├── models/                - Serialized `.pkl` trained ML models (Random Forest, Vectorizers)
  ├── scripts/               - Automation scripts for merging datasets and sanity checks
  ├── app.py                 - Core Flask Router, ML Inference, and Fusion Engine logic
  └── requirements.txt       - Python backend dependencies
frontend/                    - Vite-powered React UI Application
  ├── src/
  │   ├── assets/            - Global styles (variables, base resets), Images/SVGs
  │   ├── components/        - UI Modules (Visualizer, Dashboard, Assessment, Landing Page)
  │   ├── App.jsx            - Core entrypoint and Routes definition
  │   └── main.jsx           - DOM rendering and Context providers
  ├── package.json           - NPM dependencies and build scripts
  └── vite.config.js         - Bundler configurations
screenshots/                 - Documentation assets and UI highlights
```

---

## ⚙️ Quickstart (Windows)

Start both servers to experience CAPE locally. 

### 1. Backend Setup (Flask + ML Models)
1. Open up a terminal and navigate to the project root:
2. Create & activate a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. Install ML and Backend dependencies:
   ```powershell
   pip install -r backend/requirements.txt
   ```
4. Start the Flask server:
   ```powershell
   python backend/app.py
   ```
*The API will start on `http://127.0.0.1:5000`.*

### 2. Frontend Setup (React + Vite)
1. Open a **new** terminal window and navigate into the `frontend` directory:
   ```powershell
   cd frontend
   ```
2. Install Node.js dependencies:
   ```powershell
   npm install
   ```
3. Boot the development server:
   ```powershell
   npm run dev
   ```
*The app will automatically open in your browser, typically at `http://localhost:5173`.*

---

## 🤝 Core Contributors

<div align="center" style="display:flex;gap:40px;flex-wrap:nowrap;justify-content:center;margin:20px 0;">

  <a href="https://github.com/Shash309" style="text-decoration:none;color:inherit;text-align:center;">
    <figure style="margin:0;">
      <img src="https://github.com/Shash309.png" width="90" height="90" alt="Shashwat Sharma" style="border-radius:50%;box-shadow:0 6px 20px rgba(11,22,39,0.12);display:block;border:2px solid #e9eef2;" />
      <figcaption style="font-weight:700;color:#0b63d6;margin-top:10px;font-size:15px;letter-spacing:0.5px;">Shashwat Sharma</figcaption>
    </figure>
  </a>

  <a href="https://github.com/SwAsTiK6937" style="text-decoration:none;color:inherit;text-align:center;">
    <figure style="margin:0;">
      <img src="https://github.com/SwAsTiK6937.png" width="90" height="90" alt="Swastik Pandey" style="border-radius:50%;box-shadow:0 6px 20px rgba(11,22,39,0.12);display:block;border:2px solid #e9eef2;" />
      <figcaption style="font-weight:700;color:#0b63d6;margin-top:10px;font-size:15px;letter-spacing:0.5px;">Swastik Pandey</figcaption>
    </figure>
  </a>

</div>

---

## 📜 License
This project is licensed under the **MIT License**.