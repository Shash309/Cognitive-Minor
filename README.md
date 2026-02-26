# 🤖 CAPE — Career Path Explorer

<p align="center">
  <img src="./assets/hero.gif" alt="CAPE Hero" width="760" style="border-radius:12px;box-shadow:0 8px 30px rgba(11,22,39,0.12)"/>
</p>

<p align="center">
  <a href="#"><img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue"/></a>
  <a href="#"><img alt="License" src="https://img.shields.io/badge/license-MIT-green"/></a>
  <a href="#"><img alt="Status" src="https://img.shields.io/badge/status-Active-brightgreen"/></a>
</p>

CAPE is an explainable career recommendation engine and path visualizer that maps student profiles (skills, education, interests) to ranked career suggestions and interactive educational roadmaps.

---

## 🚀 Highlights
- **Explainable AI**: Random Forest model with feature importance insights.
- **Interactive Career Journey**: A modern, mind-map styled visualizer with **Path Tracing**.
- **Roadmap Timeline**: Step-by-step guidance generated dynamically based on selected career goals.
- **End-to-End Pipeline**: Includes data preprocessing, feature engineering, and model training.
- **Modern UI**: Bento-box inspired React frontend with smooth animations and vibrant aesthetics.

---

## 🖼️ Latest Improvements: Visualizer 2.0

<div align="center" style="display:flex;flex-wrap:wrap;gap:18px;justify-content:center;max-width:1200px;margin:0 auto;">

  <figure style="width:100%;max-width:800px;margin:0;text-align:center;font-family:system-ui,Segoe UI,Helvetica,Arial,sans-serif;">
    <a href="./screenshots/Visualizer.png" style="display:block;text-decoration:none;color:inherit;">
      <img src="./screenshots/Visualizer.png" alt="Improved Visualizer"
        style="width:100%;height:auto;border-radius:12px;border:1px solid #e9eef2;box-shadow:0 12px 40px rgba(12,20,32,0.1);transition:transform .18s ease;display:block;" />
    </a>
    <figcaption style="font-size:14px;color:#556;line-height:1.2;margin-top:12px;font-weight:600;">
      Visualizer 2.0 — Interactive Path Mapping with Roadmap Timeline
    </figcaption>
  </figure>

</div>

The visualizer now features:
- **Directional Path Tracing**: Click any stage to highlight the entire journey from root to career.
- **Animated Dotted Connections**: Visual guidance showing the flow of progression.
- **Path Glow**: Selected nodes emit a high-intensity glow for clear focus.
- **Dynamic Timeline**: A step-by-step roadmap at the bottom providing clear milestones.

---

## 📁 Project Structure

```
.vscode/                     - editor settings
backend/                     - Python backend (Flask)
  ├── data/                  - datasets (colleges.csv, specialized datasets)
  ├── models/                - trained .pkl artifacts (Random Forest, Vectorizers)
  ├── scripts/               - utility scripts (merging, testing, shape checking)
  ├── app.py                 - Flask API entry point
  └── requirements.txt       - Backend dependencies
frontend/                    - Vite/React app (UI)
  ├── src/
  │   ├── components/        - UI components (Visualizer, Dashboard, etc.)
  │   └── assets/            - styling and images
  ├── package.json
  └── README.md
screenshots/                 - documentation images
```

---

## ⚙️ Quickstart (Windows)

### 1. Backend Setup
1. Create & activate venv:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   pip install -r backend/requirements.txt
   ```
3. Start Flask server:
   ```powershell
   python backend/app.py
   ```

### 2. Frontend Setup
1. Install & build:
   ```powershell
   cd frontend
   npm install
   ```
2. Start Dev server:
   ```powershell
   npm run dev
   ```

---

## 🤝 Contributing

- Contributions are welcome — please open an issue or PR.
- Follow the established design language for UI components.

### ✨ Core Contributors

<div align="center" style="display:flex;gap:28px;flex-wrap:nowrap;align-items:flex-start;justify-content:center;margin:16px 0;">

  <a href="https://github.com/vansh070605" style="text-decoration:none;color:inherit;text-align:center;">
    <figure style="margin:0;">
      <img src="https://github.com/vansh070605.png" width="80" height="80" alt="Vansh Agrawal" style="border-radius:50%;box-shadow:0 6px 18px rgba(11,22,39,0.08);display:block;" />
      <figcaption style="font-weight:600;color:#0b63d6;margin-top:8px;font-size:14px;">Vansh Agrawal</figcaption>
    </figure>
  </a>

  <a href="https://github.com/VanshRajput-dev" style="text-decoration:none;color:inherit;text-align:center;">
    <figure style="margin:0;">
      <img src="https://github.com/VanshRajput-dev.png" width="80" height="80" alt="Vansh.C" style="border-radius:50%;box-shadow:0 6px 18px rgba(11,22,39,0.08);display:block;" />
      <figcaption style="font-weight:600;color:#0b63d6;margin-top:8px;font-size:14px;">Vansh.C</figcaption>
    </figure>
  </a>

  <a href="https://github.com/SwAsTiK6937" style="text-decoration:none;color:inherit;text-align:center;">
    <figure style="margin:0;">
      <img src="https://github.com/SwAsTiK6937.png" width="80" height="80" alt="Swastik Pandey" style="border-radius:50%;box-shadow:0 6px 18px rgba(11,22,39,0.08);display:block;" />
      <figcaption style="font-weight:600;color:#0b63d6;margin-top:8px;font-size:14px;">Swastik Pandey</figcaption>
    </figure>
  </a>

  <a href="https://github.com/Shash309" style="text-decoration:none;color:inherit;text-align:center;">
    <figure style="margin:0;">
      <img src="https://github.com/Shash309.png" width="80" height="80" alt="Shashwat Sharma" style="border-radius:50%;box-shadow:0 6px 18px rgba(11,22,39,0.08);display:block;" />
      <figcaption style="font-weight:600;color:#0b63d6;margin-top:8px;font-size:14px;">Shashwat Sharma</figcaption>
    </figure>
  </a>

</div>

---

## 📜 License
Licensed under MIT.
