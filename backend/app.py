import os
import warnings
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.exceptions import InconsistentVersionWarning
from sentence_transformers import SentenceTransformer

# ================== Setup Flask ==================
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

app = Flask(__name__)
CORS(app)  # allow frontend access

BASE = os.path.dirname(__file__)

# ================== Load Quiz ML Model ==================
try:
    quiz_model = joblib.load(os.path.join(BASE, "models", "career_1200_model.pkl"))
    quiz_vectorizer = joblib.load(os.path.join(BASE, "models", "quiz_vectorizer.pkl"))
    quiz_encoder = joblib.load(os.path.join(BASE, "models", "quiz_label_encoder.pkl"))
    print("✅ Quiz model loaded")
except Exception as e:
    print("⚠️ Could not load quiz model:", e)
    quiz_model = quiz_vectorizer = quiz_encoder = None

print("Loading sentence-transformer (may download model first time)...")
emb_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded. Embedding dim:", emb_model.get_sentence_embedding_dimension())

# ================== Load Colleges Dataset ==================
try:
    colleges = pd.read_csv(os.path.join(BASE, "data", "colleges.csv"))
    colleges.columns = [c.strip().lower() for c in colleges.columns]  # normalize headers
    print("✅ Colleges dataset loaded with", len(colleges), "rows")
except Exception as e:
    print("⚠️ Could not load colleges dataset:", e)
    colleges = pd.DataFrame()

# ================== ML Routes ==================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    if quiz_model is None or quiz_vectorizer is None or quiz_encoder is None or emb_model is None:
        return jsonify({"error": "Model or required assets not available"}), 500

    data = request.json
    text_input = data.get("features")  # expecting single string
    if not text_input:
        return jsonify({"error": "No text provided"}), 400

    try:
        # Embedding + TF-IDF features
        emb = emb_model.encode([text_input], convert_to_numpy=True)  # (1, EMB_DIM)
        tfidf = quiz_vectorizer.transform([text_input]).toarray()    # (1, TFIDF_DIM)

        print("embedding shape:", emb.shape, "tfidf shape:", tfidf.shape)

        X = np.hstack([emb, tfidf])

        # Check model input dimensions
        if hasattr(quiz_model, "n_features_in_"):
            expected = quiz_model.n_features_in_
            if X.shape[1] != expected:
                return jsonify({
                    "error": f"Feature shape mismatch: model expects {expected}, got {X.shape[1]}"
                }), 500

        # Predict career
        pred = quiz_model.predict(X)[0]
        career = quiz_encoder.inverse_transform([pred])[0]
        return jsonify({"career": career})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ================== Basic Colleges Route ==================
@app.route("/colleges", methods=["POST"])
def get_colleges():
    if colleges.empty:
        return jsonify({"error": "Colleges dataset not available"}), 500

    data = request.json
    career = data.get("career")
    state = data.get("state")

    if not career or not state:
        return jsonify({"error": "Career and State required"}), 400

    try:
        filtered = colleges[
            (colleges["state"].str.lower() == state.lower()) &
            (colleges["field"].str.lower() == career.lower())
        ]

        if filtered.empty:
            return jsonify({"error": "No colleges found"}), 404

        top5 = filtered.sort_values("ranking").head(5)
        return jsonify(top5.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": f"College lookup failed: {str(e)}"}), 500

# ================== Advanced Colleges Routes ==================
@app.route("/api/states", methods=["GET"])
def get_states():
    """Return list of unique states from colleges dataset"""
    if colleges.empty or "state" not in colleges.columns:
        return jsonify([]), 200
    states = sorted(colleges["state"].dropna().unique().tolist())
    return jsonify(states), 200


@app.route("/api/recommend", methods=["POST"])
def recommend():
    """Return college recommendations for a given state"""
    if colleges.empty:
        return jsonify({"error": "No college data available"}), 500

    data = request.get_json()
    if not data or "state" not in data:
        return jsonify({"error": "Missing state parameter"}), 400

    state = data.get("state")
    state_colleges = colleges[colleges["state"].str.lower() == state.lower()]

    if state_colleges.empty:
        return jsonify([]), 200

    def get_best_ranking(rankings):
        valid = [r for r in rankings if pd.notnull(r)]
        return min(valid) if valid else np.inf

    # Build aggregation dict dynamically
    agg_dict = {
        'city': ('city', 'first'),
        'rankings': ('ranking', list),
        'fields': ('field', list),
        'scores': ('score', list)
    }

    if 'website' in state_colleges.columns:  # ✅ only add if exists
        agg_dict['website'] = ('website', 'first')

    grouped = state_colleges.groupby('college_name').agg(**agg_dict).reset_index()
    grouped['best_ranking'] = grouped['rankings'].apply(get_best_ranking)
    grouped.sort_values('best_ranking', inplace=True)

    colleges_list = []
    for _, college in grouped.iterrows():
        college_info = {
            "college_name": college["college_name"],
            "city": college["city"],
            "website": college["website"] if "website" in grouped.columns else None,
            "rankings": [
                {
                    "field": college['fields'][i],
                    "ranking": None if pd.isna(college['rankings'][i]) else int(college['rankings'][i]),
                    "score": None if pd.isna(college['scores'][i]) else float(college['scores'][i])
                } for i in range(len(college['fields']))
            ]
        }
        colleges_list.append(college_info)

    return jsonify(colleges_list), 200

# ================== Run ==================
if __name__ == "__main__":
    app.run(debug=True, port=5000)
