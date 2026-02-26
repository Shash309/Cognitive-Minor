# ⚙️ CAPE Backend — Logic & AI

This is the backend service for **CAPE (Career Path Explorer)**. It handles data processing, model inference, and provides the API for the frontend visualizer.

## ✨ Key Features
- **API service**: Flask-based REST API for serving career recommendations.
- **Explainable Models**: Random Forest classifier providing feature importance.
- **Data Pipeline**: Automated processing of student profiles and academic data.
- **Model Storage**: Pre-trained artifacts for quick inference.

## 📁 Directory Structure
- `data/`: Contains raw and processed CSV datasets.
- `models/`: Pickled model files (`.pkl`).
- `scripts/`: Maintenance and data cleaning scripts.
- `app.py`: Main Flask application.

## 🚀 Getting Started

1. **Setup Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Or .venv\Scripts\Activate on Windows
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the server**:
   ```bash
   python app.py
   ```

The server will start on `http://127.0.0.1:5000` by default.
