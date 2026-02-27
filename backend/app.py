import os
import json
import warnings
from datetime import datetime

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

DATA_DIR = os.path.join(BASE, "data")
PSYCH_PATH = os.path.join(DATA_DIR, "psych_profiles.json")
QUIZ_HISTORY_PATH = os.path.join(DATA_DIR, "quiz_history.json")
CAREER_FUSED_RESULTS_PATH = os.path.join(DATA_DIR, "career_fused_results.json")
USERS_PATH = os.path.join(DATA_DIR, "users.json")


def _load_json_db(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_json_db(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


psych_db = _load_json_db(PSYCH_PATH)
quiz_db = _load_json_db(QUIZ_HISTORY_PATH)
fused_results_db = _load_json_db(CAREER_FUSED_RESULTS_PATH)
users_db = _load_json_db(USERS_PATH)

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
    colleges = pd.read_csv(os.path.join(DATA_DIR, "colleges.csv"))
    colleges.columns = [c.strip().lower() for c in colleges.columns]  # normalize headers
    print("✅ Colleges dataset loaded with", len(colleges), "rows")
except Exception as e:
    print("⚠️ Could not load colleges dataset:", e)
    colleges = pd.DataFrame()

PSYCH_QUESTION_META = {
    # Openness / creativity
    "O1": {"dimension": "openness", "reverse": False},
    "O2": {"dimension": "openness", "reverse": False},
    "O3": {"dimension": "creativity_preference", "reverse": False},
    # Conscientiousness / structure
    "C1": {"dimension": "conscientiousness", "reverse": False},
    "C2": {"dimension": "conscientiousness", "reverse": False},
    "C3": {"dimension": "structure_preference", "reverse": False},
    # Extraversion / individual contributor
    "E1": {"dimension": "extraversion", "reverse": False},
    "E2": {"dimension": "extraversion", "reverse": False},
    "E3": {"dimension": "individual_contributor", "reverse": True},
    # Agreeableness / leadership
    "A1": {"dimension": "agreeableness", "reverse": False},
    "A2": {"dimension": "agreeableness", "reverse": False},
    "A3": {"dimension": "leadership_index", "reverse": False},
    # Neuroticism / stress
    "N1": {"dimension": "neuroticism", "reverse": True},
    "N2": {"dimension": "neuroticism", "reverse": False},
    "N3": {"dimension": "stress_tolerance", "reverse": False},
    # Decision-making
    "D1": {"dimension": "analytical_thinking", "reverse": False},
    "D2": {"dimension": "analytical_thinking", "reverse": False},
    "D3": {"dimension": "intuitive_preference", "reverse": False},
    # Risk
    "R1": {"dimension": "risk_tolerance", "reverse": False},
    "R2": {"dimension": "risk_tolerance", "reverse": True},
    # Motivation
    "M1": {"dimension": "intrinsic_motivation", "reverse": False},
    "M2": {"dimension": "intrinsic_motivation", "reverse": False},
    "M3": {"dimension": "extrinsic_motivation", "reverse": False},
    # Leadership vs IC
    "L1": {"dimension": "leadership_index", "reverse": False},
    "L2": {"dimension": "leadership_index", "reverse": False},
    "L3": {"dimension": "individual_contributor", "reverse": False},
    # Creativity vs structure
    "CS1": {"dimension": "creativity_preference", "reverse": False},
    "CS2": {"dimension": "structure_preference", "reverse": False},
    # Scenario / behavior
    "S1": {"dimension": "stress_tolerance", "reverse": False},
    "S2": {"dimension": "analytical_thinking", "reverse": False},
    "S3": {"dimension": "openness", "reverse": False},
}

CAREER_TRAIT_WEIGHTS = {
    "Data Scientist": {
        "analytical_thinking": 0.45,
        "openness": 0.2,
        "conscientiousness": 0.15,
        "extraversion": -0.05,
        "risk_tolerance": 0.1,
    },
    "Researcher": {
        "analytical_thinking": 0.35,
        "openness": 0.25,
        "introversion": 0.1,
        "stress_tolerance": 0.1,
    },
    "Software Engineer": {
        "analytical_thinking": 0.35,
        "conscientiousness": 0.25,
        "structure_preference": 0.15,
        "risk_tolerance": 0.05,
    },
    "Manager": {
        "leadership_index": 0.35,
        "extraversion": 0.25,
        "agreeableness": 0.15,
        "conscientiousness": 0.1,
    },
    "Entrepreneur": {
        "risk_tolerance": 0.3,
        "openness": 0.2,
        "leadership_index": 0.2,
        "stress_tolerance": 0.15,
    },
    "Designer": {
        "creativity_preference": 0.35,
        "openness": 0.25,
        "agreeableness": 0.1,
    },
    "Psychologist": {
        "agreeableness": 0.3,
        "openness": 0.2,
        "intrinsic_motivation": 0.15,
        "analytical_thinking": 0.1,
    },
    "Civil Servant": {
        "conscientiousness": 0.3,
        "stress_tolerance": 0.2,
        "agreeableness": 0.15,
        "leadership_index": 0.1,
    },
    "Doctor": {
        "conscientiousness": 0.3,
        "stress_tolerance": 0.25,
        "agreeableness": 0.15,
        "intrinsic_motivation": 0.1,
    },
    "Teacher": {
        "agreeableness": 0.25,
        "intrinsic_motivation": 0.25,
        "extraversion": 0.1,
    },
    "Artist": {
        "creativity_preference": 0.35,
        "openness": 0.3,
        "risk_tolerance": 0.1,
    },
}

# ================== Fusion Weights (Configurable) ==================
QUIZ_FUSION_WEIGHT = 0.5
PSYCH_FUSION_WEIGHT = 0.5


def _apply_stream_boost(quiz_scores: dict | None, stream: str | None):
    """
    Small, subtle adjustment (max +5) to quiz-based scores based on academic stream.
    Does not change underlying ML probabilities, only post-processed scores.
    """
    if not quiz_scores or not stream:
        return quiz_scores

    stream_key = str(stream).strip().lower()

    boost_map = {
        "science": ["Data Scientist", "Software Engineer", "Researcher", "Doctor"],
        "commerce": ["Manager", "Entrepreneur"],
        "arts": ["Designer", "Psychologist", "Civil Servant", "Teacher", "Artist"],
    }

    target_careers = boost_map.get(stream_key, [])
    if not target_careers:
        return quiz_scores

    for career in target_careers:
        if career in quiz_scores:
            base = float(quiz_scores[career])
            quiz_scores[career] = max(0.0, min(100.0, base + 5.0))

    return quiz_scores


def _touch_user_profile(email: str | None, name: str | None = None):
    """Create or update a lightweight user profile for profile page use."""
    if not email:
        return

    now = datetime.utcnow().isoformat() + "Z"
    profile = users_db.get(email) or {
        "email": email,
        "created_at": now,
    }
    if name:
        profile["name"] = name
    profile.setdefault("created_at", now)
    profile["last_login"] = now

    users_db[email] = profile
    _save_json_db(USERS_PATH, users_db)


def _compute_psych_profile(responses: dict) -> dict:
    """Convert raw Likert responses into normalized 0–100 trait scores."""
    trait_sums = {}
    trait_counts = {}

    for qid, raw_val in (responses or {}).items():
        meta = PSYCH_QUESTION_META.get(qid)
        if not meta:
            continue
        try:
            val = int(raw_val)
        except (TypeError, ValueError):
            continue
        if not 1 <= val <= 5:
            continue
        if meta.get("reverse"):
            val = 6 - val
        dim = meta["dimension"]
        trait_sums[dim] = trait_sums.get(dim, 0.0) + float(val)
        trait_counts[dim] = trait_counts.get(dim, 0) + 1

    profile = {}
    for dim, total in trait_sums.items():
        count = trait_counts.get(dim, 0)
        if count == 0:
            continue
        min_raw = 1 * count
        max_raw = 5 * count
        if max_raw == min_raw:
            norm = 50.0
        else:
            norm = (total - min_raw) / float(max_raw - min_raw) * 100.0
        profile[dim] = max(0.0, min(100.0, norm))

    # Ensure all primary traits exist so the UI radar chart remains stable
    for key in [
        "openness",
        "conscientiousness",
        "extraversion",
        "agreeableness",
        "neuroticism",
        "analytical_thinking",
        "risk_tolerance",
        "leadership_index",
        "stress_tolerance",
    ]:
        profile.setdefault(key, 0.0)

    return profile


def _derive_decision_style(profile: dict) -> str:
    analytical = profile.get("analytical_thinking", 50.0)
    intuitive = profile.get("intuitive_preference", 50.0)
    if analytical >= 70 and intuitive <= 55:
        return "Strongly analytical"
    if analytical >= 60 and intuitive >= 60:
        return "Balanced analytical & intuitive"
    if intuitive >= 70 and analytical <= 55:
        return "Strongly intuitive"
    return "Flexible / context-dependent"


def _build_dominant_traits(profile: dict) -> list:
    display_names = {
        "openness": "Openness to Experience",
        "conscientiousness": "Conscientiousness",
        "extraversion": "Extraversion",
        "agreeableness": "Agreeableness",
        "neuroticism": "Neuroticism",
        "analytical_thinking": "Analytical thinking",
        "risk_tolerance": "Risk tolerance",
        "stress_tolerance": "Stress tolerance",
        "intrinsic_motivation": "Intrinsic motivation",
        "extrinsic_motivation": "Extrinsic motivation",
        "leadership_index": "Leadership tendency",
        "individual_contributor": "Individual contributor preference",
        "creativity_preference": "Creativity preference",
        "structure_preference": "Structure preference",
        "intuitive_preference": "Intuitive decision-making",
    }
    items = [
        {
            "name": k,
            "display_name": display_names.get(k, k),
            "score": float(v),
        }
        for k, v in profile.items()
    ]
    items.sort(key=lambda x: x["score"], reverse=True)
    return items


def _compute_stability(prev_profile: dict | None, current_profile: dict | None):
    """Cosine-based stability index between two profiles (0–1) plus label."""
    if not prev_profile or not current_profile:
        return None, "First assessment"

    keys = sorted(set(prev_profile.keys()) | set(current_profile.keys()))
    if not keys:
        return None, "Insufficient data"

    v1 = np.array([float(prev_profile.get(k, 0.0)) for k in keys], dtype=float)
    v2 = np.array([float(current_profile.get(k, 0.0)) for k in keys], dtype=float)

    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 == 0.0 or n2 == 0.0:
        return None, "Insufficient data"

    cos_sim = float(np.dot(v1, v2) / (n1 * n2))
    cos_sim = max(-1.0, min(1.0, cos_sim))
    index = (cos_sim + 1.0) / 2.0  # map -1..1 to 0..1

    if index >= 0.8:
        label = "Highly Stable"
    elif index >= 0.5:
        label = "Moderately Changing"
    else:
        label = "Significant Shift"

    return index, label


def _get_quiz_snapshot(email: str):
    if not email:
        return None
    return quiz_db.get(email)


def _get_quiz_context(email: str):
    """
    Return (quiz_scores, academic_percent, skills_vector) for a user.
    Backwards compatible with older quiz_history records.
    """
    snapshot = _get_quiz_snapshot(email)
    if not snapshot:
        return None, None, {}

    quiz_scores = snapshot.get("quiz_scores")

    academic_percent = snapshot.get("academic_percent")
    if academic_percent is not None:
        try:
            academic_percent = float(academic_percent)
        except (TypeError, ValueError):
            academic_percent = None

    skills_vector = snapshot.get("skills_vector") or snapshot.get("skills") or {}

    return quiz_scores, academic_percent, skills_vector


def _get_latest_psych_scores(email: str | None):
    if not email:
        return None
    history = psych_db.get(email) or []
    latest = history[0] if history else None
    if not latest:
        return None
    return latest.get("psych_scores")


def _psych_alignment_for_career(profile: dict, career: str) -> float:
    """
    Compute pure psychological alignment (0–100) for a single career
    based only on trait weights and the psychological profile.
    """
    weights = CAREER_TRAIT_WEIGHTS.get(career) or {}
    if not weights:
        return 50.0

    total_weight = sum(abs(w) for w in weights.values()) or 1.0

    score = 0.0
    for trait, weight in weights.items():
        if trait == "introversion":
            base = 100.0 - profile.get("extraversion", 50.0)
        else:
            base = profile.get(trait, 50.0)
        centered = (base - 50.0) / 50.0  # -1 .. 1
        score += weight * centered

    normalized = (score / total_weight + 1.0) / 2.0  # 0..1
    return max(0.0, min(100.0, normalized * 100.0))


def _compute_psych_career_scores(profile: dict) -> dict:
    """
    Return pure psychological alignment scores (0–100) for all careers.
    """
    scores: dict[str, float] = {}
    for career in CAREER_TRAIT_WEIGHTS.keys():
        scores[career] = round(_psych_alignment_for_career(profile, career))
    return scores


def _fuse_career_scores(quiz_scores, psych_scores, academic_percent=None, skills_vector=None):
    """
    Fusion Engine: combine quiz_scores and psych_scores into final unified rankings.

    Final score per career:
        final = w_q * quiz_score + w_p * psych_score

    Where:
      - If only quiz_scores present => w_q = 1, w_p = 0
      - If only psych_scores present => w_q = 0, w_p = 1
      - If both present => base weights QUIZ_FUSION_WEIGHT / PSYCH_FUSION_WEIGHT,
        normalized to sum to 1.
    """
    have_quiz = bool(quiz_scores)
    have_psych = bool(psych_scores)

    if not have_quiz and not have_psych:
        return {"career_rankings": []}

    if have_quiz and not have_psych:
        w_q, w_p = 1.0, 0.0
    elif have_psych and not have_quiz:
        w_q, w_p = 0.0, 1.0
    else:
        total = float(QUIZ_FUSION_WEIGHT + PSYCH_FUSION_WEIGHT) or 1.0
        w_q = QUIZ_FUSION_WEIGHT / total
        w_p = PSYCH_FUSION_WEIGHT / total

    careers = set()
    if isinstance(quiz_scores, dict):
        careers.update(quiz_scores.keys())
    if isinstance(psych_scores, dict):
        careers.update(psych_scores.keys())

    rankings = []
    for career in careers:
        q = float(quiz_scores.get(career)) if have_quiz and career in quiz_scores else None
        p = float(psych_scores.get(career)) if have_psych and career in psych_scores else None

        if q is None and p is None:
            continue

        if not have_quiz:
            final_score = p
        elif not have_psych:
            final_score = q
        else:
            final_score = (w_q * q if q is not None else 0.0) + (w_p * p if p is not None else 0.0)

        rankings.append(
            {
                "career": career,
                "final_score": float(final_score),
                "quiz_component": q,
                "psych_component": p,
            }
        )

    rankings.sort(key=lambda x: x["final_score"], reverse=True)

    return {
        "career_rankings": rankings,
    }


def _update_fused_results(
    email: str | None,
    quiz_scores,
    psych_scores,
    fusion_result: dict,
    academic_percent=None,
    skills_vector=None,
):
    if not email:
        return

    rankings = fusion_result.get("career_rankings") or []
    final_scores = {item["career"]: item["final_score"] for item in rankings}

    fused_results_db[email] = {
        "quiz_scores": quiz_scores or {},
        "psych_scores": psych_scores or {},
        "final_scores": final_scores,
        "academic_percent": academic_percent,
        "skills_vector": skills_vector or {},
        "last_updated": datetime.utcnow().isoformat() + "Z",
    }
    _save_json_db(CAREER_FUSED_RESULTS_PATH, fused_results_db)


def _compute_career_alignment(profile: dict, email: str | None):
    quiz_snapshot = _get_quiz_snapshot(email)
    academic_percent = None
    skills_vector = {}

    if quiz_snapshot:
        academic_percent = quiz_snapshot.get("academic_percent")
        if academic_percent is not None:
            try:
                academic_percent = float(academic_percent)
            except (TypeError, ValueError):
                academic_percent = None
        skills_vector = quiz_snapshot.get("skills_vector") or quiz_snapshot.get("skills") or {}

    if academic_percent is None:
        academic_percent = 70.0  # reasonable neutral baseline

    def academic_component(_career: str) -> float:
        return max(0.0, min(100.0, academic_percent))

    def skills_component(career: str) -> float:
        # Very lightweight heuristic: if we have no structured skills, return neutral.
        if not skills_vector:
            return 70.0
        # Each career can optionally define expected skills. For now use broad tags.
        expected = {
            "Data Scientist": ["analysis", "maths", "programming", "research"],
            "Researcher": ["research", "reading", "analysis"],
            "Software Engineer": ["programming", "coding", "problemSolving"],
            "Manager": ["leadership", "communication", "organizingEvents"],
            "Entrepreneur": ["entrepreneur", "leadership", "risk"],
            "Designer": ["design", "designThinking", "creativity"],
            "Psychologist": ["psychology", "helpingOthers", "reading"],
            "Civil Servant": ["politicalScience", "history", "reading"],
            "Doctor": ["biology", "chemistry"],
            "Teacher": ["presentation", "communication"],
            "Artist": ["fineArts", "drawing"],
        }.get(career, [])

        if not expected:
            return 70.0

        total = 0.0
        count = 0
        for key in expected:
            val = skills_vector.get(key)
            if val is None:
                continue
            total += float(val)
            count += 1
        if count == 0:
            return 70.0
        return max(0.0, min(100.0, (total / count) * 100.0))

    results = []
    for career in CAREER_TRAIT_WEIGHTS.keys():
        a = academic_component(career)
        s = skills_component(career)
        p = _psych_alignment_for_career(profile, career)
        overall = 0.4 * a + 0.3 * s + 0.3 * p

        # Simple skill-gap suggestion from psychological traits
        gaps = []
        for trait, weight in (CAREER_TRAIT_WEIGHTS.get(career) or {}).items():
            if trait == "introversion":
                trait_name = "extraversion"
                current_val = 100.0 - profile.get("extraversion", 50.0)
            else:
                trait_name = trait
                current_val = profile.get(trait_name, 50.0)
            if weight > 0 and current_val < 65.0:
                desired = min(95.0, 75.0 + (weight * 20.0))
                gaps.append(
                    {
                        "name": trait_name,
                        "display_name": trait_name.replace("_", " ").title(),
                        "current": float(current_val),
                        "desired": float(desired),
                    }
                )

        results.append(
            {
                "career": career,
                "overall_score": overall,
                "academic_component": a,
                "skills_component": s,
                "psych_component": p,
                "skill_gaps": gaps,
            }
        )

    results.sort(key=lambda x: x["overall_score"], reverse=True)

    # Explainability for top career
    if results:
        top = results[0]
        top_career = top["career"]
        weights = CAREER_TRAIT_WEIGHTS.get(top_career) or {}
        contributions = []
        for trait, weight in weights.items():
            if trait == "introversion":
                trait_name = "extraversion"
                base_val = 100.0 - profile.get("extraversion", 50.0)
            else:
                trait_name = trait
                base_val = profile.get(trait_name, 50.0)
            centered = (base_val - 50.0) / 50.0
            contrib = abs(weight * centered)
            contributions.append(
                {
                    "name": trait_name,
                    "display_name": trait_name.replace("_", " ").title(),
                    "importance": contrib,
                    "score": float(base_val),
                }
            )
        contributions.sort(key=lambda x: x["importance"], reverse=True)
        top["top_traits"] = contributions[:3]

    return results


# ================== ML Routes ==================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/psych-assessment", methods=["GET"])
def get_psych_assessment():
    """Return latest psychological profile and history for a user."""
    user_email = request.args.get("user_email")
    if not user_email:
        return jsonify({"error": "user_email is required"}), 400

    history = psych_db.get(user_email) or []
    latest = history[0] if history else None

    response = {
        "profile": latest.get("profile") if latest else None,
        "decision_style": latest.get("decision_style") if latest else None,
        "dominant_traits": latest.get("dominant_traits") if latest else None,
        "career_matches": latest.get("career_matches") if latest else None,
        "stability_index": latest.get("stability_index") if latest else None,
        "stability_label": latest.get("stability_label") if latest else None,
        "history": [
            {
                "completed_at": item.get("completed_at"),
                "top_career": (item.get("career_matches") or [{}])[0].get("career")
                if item.get("career_matches")
                else None,
            }
            for item in history
        ],
    }
    return jsonify(response), 200


@app.route("/api/psych-assessment", methods=["POST"])
def post_psych_assessment():
    """Compute psychological profile, store it, and generate explainable career recommendations."""
    data = request.json or {}
    user_email = data.get("user_email")
    responses = data.get("responses") or {}
    retake_reason = data.get("retake_reason")

    if not user_email:
        return jsonify({"error": "user_email is required"}), 400
    if not isinstance(responses, dict) or not responses:
        return jsonify({"error": "responses must be a non-empty object"}), 400

    profile = _compute_psych_profile(responses)
    decision_style = _derive_decision_style(profile)
    dominant_traits = _build_dominant_traits(profile)
    career_matches = _compute_career_alignment(profile, user_email)
    psych_scores = _compute_psych_career_scores(profile)

    user_history = psych_db.get(user_email, [])
    previous_profile = user_history[0].get("profile") if user_history else None
    stability_index, stability_label = _compute_stability(previous_profile, profile)

    record = {
        "completed_at": datetime.utcnow().isoformat() + "Z",
        "profile": profile,
        "decision_style": decision_style,
        "dominant_traits": dominant_traits,
        "career_matches": career_matches,
        "psych_scores": psych_scores,
        "stability_index": stability_index,
        "stability_label": stability_label,
        "retake_reason": retake_reason,
    }

    user_history.insert(0, record)
    # keep recent N records
    psych_db[user_email] = user_history[:10]
    _save_json_db(PSYCH_PATH, psych_db)

    # Compute/update fused career scores for this user
    quiz_scores, academic_percent, skills_vector = _get_quiz_context(user_email)
    fusion_result = _fuse_career_scores(quiz_scores, psych_scores, academic_percent, skills_vector)
    _update_fused_results(
        user_email,
        quiz_scores=quiz_scores,
        psych_scores=psych_scores,
        fusion_result=fusion_result,
        academic_percent=academic_percent,
        skills_vector=skills_vector,
    )

    response = {
        "profile": profile,
        "decision_style": decision_style,
        "dominant_traits": dominant_traits,
        "career_matches": career_matches,
        "psych_scores": psych_scores,
        "stability_index": stability_index,
        "stability_label": stability_label,
        "career_rankings": fusion_result.get("career_rankings"),
        "history": [
            {
                "completed_at": item.get("completed_at"),
                "top_career": (item.get("career_matches") or [{}])[0].get("career")
                if item.get("career_matches")
                else None,
            }
            for item in psych_db.get(user_email, [])
        ],
    }
    return jsonify(response), 200


def _predict_career_from_text(text_input: str):
    if quiz_model is None or quiz_vectorizer is None or quiz_encoder is None or emb_model is None:
        raise RuntimeError("Model or required assets not available")

    # Embedding + TF-IDF features
    emb = emb_model.encode([text_input], convert_to_numpy=True)  # (1, EMB_DIM)
    tfidf = quiz_vectorizer.transform([text_input]).toarray()    # (1, TFIDF_DIM)

    X = np.hstack([emb, tfidf])

    # Check model input dimensions
    if hasattr(quiz_model, "n_features_in_"):
        expected = quiz_model.n_features_in_
        if X.shape[1] != expected:
            raise ValueError(f"Feature shape mismatch: model expects {expected}, got {X.shape[1]}")

    pred = quiz_model.predict(X)[0]
    career = quiz_encoder.inverse_transform([pred])[0]
    return career


def _compute_quiz_career_scores(text_input: str) -> dict:
    """
    Use the ML quiz model to produce a probability-style score (0–100)
    for every career label instead of a single prediction.
    """
    if quiz_model is None or quiz_vectorizer is None or quiz_encoder is None or emb_model is None:
        raise RuntimeError("Model or required assets not available")

    emb = emb_model.encode([text_input], convert_to_numpy=True)
    tfidf = quiz_vectorizer.transform([text_input]).toarray()
    X = np.hstack([emb, tfidf])

    if hasattr(quiz_model, "n_features_in_"):
        expected = quiz_model.n_features_in_
        if X.shape[1] != expected:
            raise ValueError(f"Feature shape mismatch: model expects {expected}, got {X.shape[1]}")

    if not hasattr(quiz_model, "predict_proba"):
        raise RuntimeError("Quiz model does not support probability estimates (predict_proba)")

    proba = quiz_model.predict_proba(X)[0]  # (n_classes,)
    careers = quiz_encoder.inverse_transform(np.arange(len(proba)))

    scores = {}
    for career, p in zip(careers, proba):
        scores[str(career)] = round(float(p) * 100.0)

    return scores


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json or {}
    text_input = data.get("features")  # expecting single string
    if not text_input:
        return jsonify({"error": "No text provided"}), 400

    try:
        career = _predict_career_from_text(text_input)
        return jsonify({"career": career})
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/quiz/submit", methods=["POST"])
def quiz_submit():
    """Store structured quiz data and return ML-based career score vector (no final decision)."""
    data = request.json or {}
    text_input = data.get("answers_text") or data.get("features")
    if not text_input:
        return jsonify({"error": "No quiz text provided"}), 400

    user_email = data.get("user_email")
    structured_answers = data.get("structured_answers") or {}
    academic_percent = data.get("academic_percent")
    stream = data.get("stream")
    user_name = data.get("user_name")

    try:
        quiz_scores = _compute_quiz_career_scores(text_input)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    # Build a lightweight skills vector from structured answers for later integration
    skills_vector = {}
    if isinstance(structured_answers, dict):
        # Flatten multi-select answers into simple presence counts
        for qid, value in structured_answers.items():
            if isinstance(value, list):
                for v in value:
                    key = str(v)
                    skills_vector[key] = min(1.0, skills_vector.get(key, 0.0) + 0.5)
            else:
                key = str(value)
                skills_vector[key] = min(1.0, skills_vector.get(key, 0.0) + 0.5)

    # Validate and normalise academic percentage if provided
    if academic_percent is not None:
        try:
            academic_percent = float(academic_percent)
        except (TypeError, ValueError):
            return jsonify({"error": "academic_percent must be a number between 0 and 100"}), 400
        if not (0.0 <= academic_percent <= 100.0):
            return jsonify({"error": "academic_percent must be between 0 and 100"}), 400

    # Apply optional, subtle stream-based boost to quiz scores
    quiz_scores = _apply_stream_boost(quiz_scores, stream)

    # Persist per-user quiz snapshot & history
    if user_email:
        _touch_user_profile(user_email, user_name)

        existing = quiz_db.get(user_email) or {}
        attempts = existing.get("attempts") or []

        # If we had a legacy single-snapshot structure, convert it into an initial attempt
        if not attempts and existing.get("quiz_scores"):
            legacy_quiz_scores = existing.get("quiz_scores") or {}
            legacy_ts = existing.get("last_taken") or datetime.utcnow().isoformat() + "Z"
            # derive top career from legacy quiz scores if available
            top_career_legacy = None
            if isinstance(legacy_quiz_scores, dict) and legacy_quiz_scores:
                top_career_legacy = max(
                    legacy_quiz_scores.items(), key=lambda kv: float(kv[1])
                )[0]
            attempts.append(
                {
                    "timestamp": legacy_ts,
                    "quiz_scores": legacy_quiz_scores,
                    "academic_percent": existing.get("academic_percent"),
                    "stream": existing.get("stream"),
                    "top_career": top_career_legacy,
                }
            )

        # New attempt based on current submission
        top_career = None
        if isinstance(quiz_scores, dict) and quiz_scores:
            top_career = max(quiz_scores.items(), key=lambda kv: float(kv[1]))[0]

        attempts.insert(
            0,
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "quiz_scores": quiz_scores,
                "academic_percent": academic_percent,
                "stream": stream,
                "top_career": top_career,
            },
        )

        # keep recent N attempts
        attempts = attempts[:20]

        snapshot = {
            "last_taken": attempts[0]["timestamp"],
            "academic_percent": academic_percent,
            "skills": skills_vector,
            "skills_vector": skills_vector,
            "quiz_scores": quiz_scores,
            "stream": stream,
            "attempts": attempts,
        }
        quiz_db[user_email] = snapshot
        _save_json_db(QUIZ_HISTORY_PATH, quiz_db)

    # Compute fused scores using latest psychological data (if any)
    psych_scores = _get_latest_psych_scores(user_email)
    fusion_result = _fuse_career_scores(quiz_scores, psych_scores, academic_percent, skills_vector)
    _update_fused_results(
        user_email,
        quiz_scores=quiz_scores,
        psych_scores=psych_scores,
        fusion_result=fusion_result,
        academic_percent=academic_percent,
        skills_vector=skills_vector,
    )

    return jsonify(
        {
            "stored": bool(user_email),
            "academic_percent": academic_percent,
            "quiz_scores": quiz_scores,
            "psych_scores": psych_scores,
            "career_rankings": fusion_result.get("career_rankings"),
        }
    )


@app.route("/api/career-results", methods=["GET"])
def career_results():
    """Unified endpoint: return fused career rankings for a user."""
    user_email = request.args.get("user_email")
    if not user_email:
        return jsonify({"error": "user_email is required"}), 400

    quiz_scores, academic_percent, skills_vector = _get_quiz_context(user_email)
    psych_scores = _get_latest_psych_scores(user_email)

    fusion_result = _fuse_career_scores(quiz_scores, psych_scores, academic_percent, skills_vector)
    _update_fused_results(
        user_email,
        quiz_scores=quiz_scores,
        psych_scores=psych_scores,
        fusion_result=fusion_result,
        academic_percent=academic_percent,
        skills_vector=skills_vector,
    )

    return jsonify(
        {
            "career_rankings": fusion_result.get("career_rankings"),
            "quiz_scores": quiz_scores,
            "psych_scores": psych_scores,
            "academic_percent": academic_percent,
            "skills_vector": skills_vector,
        }
    )


@app.route("/api/quiz-history", methods=["GET"])
def get_quiz_history():
    """Return all quiz attempts for a user, newest first."""
    user_email = request.args.get("user_email")
    if not user_email:
        return jsonify({"error": "user_email is required"}), 400

    snapshot = quiz_db.get(user_email) or {}
    attempts = snapshot.get("attempts") or []

    # Ensure newest first ordering by timestamp if present
    def _ts(a):
        return a.get("timestamp") or ""

    attempts_sorted = sorted(attempts, key=_ts, reverse=True)

    return jsonify(
        {
            "attempts": attempts_sorted,
        }
    ), 200


@app.route("/api/profile", methods=["GET"])
def get_profile():
    """Return personal info plus latest quiz, psych, and fused results for a user."""
    user_email = request.args.get("user_email")
    if not user_email:
        return jsonify({"error": "user_email is required"}), 400

    user = users_db.get(user_email) or {
        "email": user_email,
    }

    quiz_snapshot = quiz_db.get(user_email) or {}
    quiz_attempts = quiz_snapshot.get("attempts") or []
    latest_quiz = quiz_attempts[0] if quiz_attempts else None

    psych_history = psych_db.get(user_email) or []
    latest_psych = psych_history[0] if psych_history else None

    fused = fused_results_db.get(user_email)
    if not fused:
        # lazily compute fused if needed
        quiz_scores, academic_percent, skills_vector = _get_quiz_context(user_email)
        psych_scores = _get_latest_psych_scores(user_email)
        fusion_result = _fuse_career_scores(quiz_scores, psych_scores, academic_percent, skills_vector)
        _update_fused_results(
            user_email,
            quiz_scores=quiz_scores,
            psych_scores=psych_scores,
            fusion_result=fusion_result,
            academic_percent=academic_percent,
            skills_vector=skills_vector,
        )
        fused = fused_results_db.get(user_email)

    # Extract a compact view of the fused ranking
    fused_rankings = None
    top_fused = None
    if fused:
        career_rankings = []
        if "final_scores" in fused and isinstance(fused["final_scores"], dict):
            for career, score in fused["final_scores"].items():
                career_rankings.append(
                    {
                        "career": career,
                        "final_score": float(score),
                    }
                )
            career_rankings.sort(key=lambda x: x["final_score"], reverse=True)
        else:
            # In case we only have career_rankings from the last fusion result
            quiz_scores, academic_percent, skills_vector = _get_quiz_context(user_email)
            psych_scores = _get_latest_psych_scores(user_email)
            fusion_result = _fuse_career_scores(quiz_scores, psych_scores, academic_percent, skills_vector)
            career_rankings = fusion_result.get("career_rankings") or []

        fused_rankings = career_rankings
        top_fused = career_rankings[0] if career_rankings else None

    return jsonify(
        {
            "user": user,
            "latest_quiz": latest_quiz,
            "latest_psych": latest_psych,
            "fused_top": top_fused,
            "fused_rankings": fused_rankings,
        }
    ), 200

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
