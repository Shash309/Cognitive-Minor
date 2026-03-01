import os
import json
import warnings
import math
from datetime import datetime
from uuid import uuid4

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
VOICE_HISTORY_PATH = os.path.join(DATA_DIR, "voice_history.json")


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

# Unified decision engine
from career_intelligence_engine import compute_final_decision
voice_db = _load_json_db(VOICE_HISTORY_PATH)


def _safe_number(val, default=None):
    """Convert value to finite float or return default."""
    try:
        f = float(val)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f

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

# ================== Voice Insight: Career Keyword Map ==================
# Keys must match CAREER_TRAIT_WEIGHTS for fusion alignment
CAREER_KEYWORD_MAP = {
    "Data Scientist": ["data", "analytics", "machine learning", "python", "statistics", "research", "algorithm", "coding", "programming", "analysis", "science"],
    "Researcher": ["research", "study", "academic", "papers", "discovery", "experiments", "science", "reading", "analysis", "investigate"],
    "Software Engineer": ["software", "coding", "programming", "developer", "engineer", "tech", "computer", "build", "code", "apps", "technology"],
    "Manager": ["manage", "lead", "team", "business", "leadership", "organize", "strategy", "corporate", "people"],
    "Entrepreneur": ["startup", "business", "own", "found", "entrepreneur", "risk", "innovate", "create", "venture"],
    "Designer": ["design", "creative", "art", "visual", "ui", "ux", "creativity", "aesthetic", "drawing"],
    "Psychologist": ["psychology", "help", "people", "mental", "counsel", "understand", "therapy", "behavior", "mind"],
    "Civil Servant": ["government", "public", "service", "policy", "civil", "administration", "society", "law"],
    "Doctor": ["doctor", "medical", "health", "patient", "medicine", "hospital", "clinical", "biology", "heal"],
    "Teacher": ["teach", "education", "students", "learning", "school", "share", "explain", "classroom"],
    "Artist": ["art", "creative", "paint", "music", "draw", "express", "artist", "design", "imagination"],
}

# Motivation / intent words (strong positive)
MOTIVATION_WORDS = [
    "passion", "love", "dream", "want", "excited", "driven", "determined", "inspired",
    "motivated", "goal", "achieve", "purpose", "meaning", "fulfill", "impact", "change",
]

# Positive sentiment words
POSITIVE_WORDS = [
    "love", "passion", "excited", "great", "amazing", "wonderful", "inspired", "fulfilling",
    "meaningful", "rewarding", "enjoy", "happy", "proud", "confident", "motivated",
]

# Negative sentiment words
NEGATIVE_WORDS = [
    "hate", "boring", "stressful", "difficult", "worried", "uncertain", "confused",
]

# ================== Fusion Weights (Configurable) ==================
QUIZ_FUSION_WEIGHT = 0.5
PSYCH_FUSION_WEIGHT = 0.5
# When voice exists: 0.4 quiz, 0.3 psych, 0.3 voice
QUIZ_FUSION_WEIGHT_3 = 0.4
PSYCH_FUSION_WEIGHT_3 = 0.3
VOICE_FUSION_WEIGHT = 0.3


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


# ================== Voice Insight: Speech-to-Text & NLP ==================
def _transcribe_audio(audio_bytes: bytes, format_hint: str = "webm") -> str:
    """Convert audio to text using SpeechRecognition. Returns empty string on failure."""
    try:
        import speech_recognition as sr
        from io import BytesIO
        from pydub import AudioSegment

        if not audio_bytes or len(audio_bytes) < 1000:
            return ""

        with BytesIO(audio_bytes) as bio:
            bio.seek(0)
            try:
                audio = AudioSegment.from_file(bio, format=format_hint if format_hint else "webm")
            except Exception:
                try:
                    audio = AudioSegment.from_file(bio, format="ogg")
                except Exception:
                    return ""

        audio = audio.set_frame_rate(16000).set_channels(1)
        raw_data = audio.raw_data
        sample_width = audio.sample_width
        frame_rate = audio.frame_rate

        recognizer = sr.Recognizer()
        audio_data = sr.AudioData(raw_data, frame_rate, sample_width)

        try:
            text = recognizer.recognize_google(audio_data, language="en-IN")
            return (text or "").strip()
        except sr.UnknownValueError:
            return ""
        except sr.RequestError:
            return ""
    except Exception as e:
        print("Voice transcribe error:", e)
        return ""


def _analyze_voice_text(text: str) -> dict:
    """Compute sentiment, motivation, confidence from transcribed text."""
    text_lower = (text or "").lower().strip()
    words = text_lower.split()
    unique_words = set(words)

    # Sentiment: -1 to 1
    pos_count = sum(1 for w in unique_words if w in POSITIVE_WORDS)
    neg_count = sum(1 for w in unique_words if w in NEGATIVE_WORDS)
    total = pos_count + neg_count
    if total == 0:
        sentiment = 0.5
    else:
        sentiment = (pos_count - neg_count) / max(total, 1)
        sentiment = max(-1.0, min(1.0, 0.5 + sentiment * 0.5))

    # Motivation: 0–100 from intent word frequency
    mot_count = sum(1 for w in words if w in MOTIVATION_WORDS)
    motivation_score = min(100.0, 30.0 + (mot_count / max(len(words), 1)) * 100)

    # Confidence: length + vocabulary richness

    vocab_ratio = len(unique_words) / max(len(words), 1) if words else 0
    confidence_score = min(100.0, 40.0 + len(words) * 0.5 + vocab_ratio * 30)

    return {
        "sentiment": _safe_number(sentiment, 0.5),
        "motivation_score": _safe_number(motivation_score, 50.0),
        "confidence_score": _safe_number(confidence_score, 50.0),
    }


def _compute_voice_career_scores(transcribed_text: str, sentiment: float, confidence: float) -> dict:
    """Compute 0–100 career alignment scores from voice transcript."""
    text_lower = (transcribed_text or "").lower()
    words = set(text_lower.split())

    scores = {}
    for career, keywords in CAREER_KEYWORD_MAP.items():
        matches = sum(1 for kw in keywords if kw in text_lower)
        raw = min(100.0, (matches / max(len(keywords), 1)) * 100) if keywords else 50.0
        factor = 0.7 + 0.3 * max(0, min(1, sentiment))
        factor *= 0.9 + 0.1 * (confidence / 100.0)
        scores[career] = round(max(0.0, min(100.0, _safe_number(raw * factor, 50.0))))

    return scores


def _get_latest_voice_scores(email: str | None):
    """Return latest voice career scores for a user."""
    if not email:
        return None
    history = voice_db.get(email) or []
    latest = history[0] if history else None
    if not latest:
        return None
    return latest.get("voice_scores")


def _save_voice_entry(email: str, transcript: str, voice_scores: dict, metadata: dict):
    """Append voice entry and keep max 10."""
    if not email:
        return
    history = voice_db.get(email) or []
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "transcript": transcript,
        "voice_scores": voice_scores,
        **metadata,
    }
    history.insert(0, entry)
    voice_db[email] = history[:10]
    _save_json_db(VOICE_HISTORY_PATH, voice_db)


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
    scores = latest.get("psych_scores")
    # Backwards compatibility: compute and persist psych_scores if missing
    if scores is None:
        profile = latest.get("profile")
        if isinstance(profile, dict):
            scores = _compute_psych_career_scores(profile)
            latest["psych_scores"] = scores
            psych_db[email] = history
            _save_json_db(PSYCH_PATH, psych_db)
    return scores


def has_completed_psych(user_email: str | None) -> bool:
    """
    Return True if the user has at least one completed psychological assessment
    stored in psych_db; False otherwise.
    """
    if not user_email:
        return False
    history = psych_db.get(user_email) or []
    return len(history) > 0


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
            base = 100.0 - _safe_number(profile.get("extraversion", 50.0), 50.0)
        else:
            base = _safe_number(profile.get(trait, 50.0), 50.0)
        centered = (base - 50.0) / 50.0  # -1 .. 1
        score += weight * centered

    normalized = (score / total_weight + 1.0) / 2.0  # 0..1
    if normalized is None or math.isnan(normalized) or math.isinf(normalized):
        normalized = 0.5

    # Map to 0–100 and slightly increase spread around 50 so psychological
    # differences have a visible impact in fusion and UI.
    alignment = max(0.0, min(100.0, normalized * 100.0))
    alignment = 50.0 + (alignment - 50.0) * 1.2
    return max(0.0, min(100.0, alignment))


def _compute_psych_career_scores(profile: dict) -> dict:
    """
    Return pure psychological alignment scores (0–100) for all careers.
    """
    scores: dict[str, float] = {}
    for career in CAREER_TRAIT_WEIGHTS.keys():
        val = _psych_alignment_for_career(profile, career)
        scores[career] = round(max(0.0, min(100.0, _safe_number(val, 50.0))))
    # Temporary debug: show a small sample in logs for verification
    sample = {k: scores[k] for k in list(scores.keys())[:3]}
    print("DEBUG_PSYCH_SCORES_SAMPLE", sample)
    return scores


def _fuse_career_scores(quiz_scores, psych_scores, academic_percent=None, skills_vector=None, voice_scores=None):
    """
    Backwards-compatible wrapper around the unified career_intelligence_engine.

    All backend routes should go through this helper so that quiz / psych / voice
    are always combined by a single mathematically-structured engine.
    """
    # academic_percent and skills_vector are currently handled outside the engine
    # (e.g., via _quiz_signals_for_career). They are passed through unchanged.
    engine_result = compute_final_decision(
        quiz_scores=quiz_scores,
        psych_scores=psych_scores,
        voice_scores=voice_scores,
    )
    return engine_result


def _update_fused_results(
    email: str | None,
    quiz_scores,
    psych_scores,
    fusion_result: dict,
    academic_percent=None,
    skills_vector=None,
    voice_scores=None,
):
    if not email:
        return

    rankings = fusion_result.get("career_rankings") or []
    final_scores = {item["career"]: item["final_score"] for item in rankings}

    fused_results_db[email] = {
        "quiz_scores": quiz_scores or {},
        "psych_scores": psych_scores or {},
        "voice_scores": voice_scores or {},
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


def _get_latest_psych_profile(email: str | None):
    """Return latest raw psychological trait profile (0–100 per trait) for a user."""
    if not email:
        return None
    history = psych_db.get(email) or []
    latest = history[0] if history else None
    if not latest:
        return None
    return latest.get("profile")


def _get_expected_skills_for_career(career: str) -> list[str]:
    """
    Return list of expected skill keys used when interpreting quiz-based signals.
    Mirrors the heuristic mapping used inside _compute_career_alignment.
    """
    mapping = {
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
    }
    return mapping.get(career, [])


def _prettify_skill_key(key: str) -> str:
    """Convert internal skill keys like 'problemSolving' into human-readable labels."""
    import re

    # Insert spaces before capitals, then title-case
    spaced = re.sub(r"(?<!^)([A-Z])", r" \1", key)
    return spaced.replace("_", " ").strip().title()


def _top_traits_for_career(psych_profile: dict | None, career: str):
    """
    For a given career, return top 3 psychological traits driving alignment,
    based on CAREER_TRAIT_WEIGHTS and the user's trait scores.
    """
    if not psych_profile:
        return []
    weights = CAREER_TRAIT_WEIGHTS.get(career) or {}
    if not weights:
        return []

    items = []
    for trait, weight in weights.items():
        if trait == "introversion":
            trait_name = "extraversion"
            score = 100.0 - float(psych_profile.get("extraversion", 50.0))
        else:
            trait_name = trait
            score = float(psych_profile.get(trait_name, 50.0))
        items.append(
            {
                "trait": trait_name,
                "weight": float(weight),
                "user_score": score,
            }
        )

    items.sort(key=lambda x: abs(x["weight"]), reverse=True)
    return items[:3]


def _quiz_signals_for_career(
    career: str,
    quiz_scores: dict | None,
    academic_percent: float | None,
    skills_vector: dict | None,
):
    """Derive simple, interpretable quiz-based signals for a career."""
    quiz_scores = quiz_scores or {}
    skills_vector = skills_vector or {}

    ml_probability = None
    top_quiz_careers = []

    if isinstance(quiz_scores, dict) and quiz_scores:
        # Scores are 0–100; convert to 0–1 probability for the selected career
        score_items = [
            (c, float(v))
            for c, v in quiz_scores.items()
            if isinstance(v, (int, float, str))
        ]
        score_items.sort(key=lambda kv: kv[1], reverse=True)
        top_quiz_careers = score_items[:3]
        for c, v in score_items:
            if c == career:
                ml_probability = max(0.0, min(1.0, v / 100.0))
                break

    # Extract top matching skills for this career
    expected = _get_expected_skills_for_career(career)
    skill_items = []
    for key in expected:
        val = skills_vector.get(key)
        if val is None:
            continue
        try:
            score = float(val)
        except (TypeError, ValueError):
            continue
        skill_items.append((key, score))

    skill_items.sort(key=lambda kv: kv[1], reverse=True)
    matched_skills = [_prettify_skill_key(k) for k, _ in skill_items[:3]]

    return {
        "ml_probability": ml_probability,
        "academic_percent": academic_percent,
        "matched_skills": matched_skills,
        "top_quiz_careers": [{"career": c, "score": s} for c, s in top_quiz_careers],
    }


def _generate_explanation(
    career: str,
    final_score: float,
    quiz_component,
    psych_component,
    quiz_contribution,
    psych_contribution,
    top_traits: list,
    quiz_signals: dict,
    confidence_score: float | None,
    alt_career: str | None,
    voice_component=None,
    voice_contribution=None,
    signal_weights: dict | None = None,
):
    """
    Construct a human-readable explanation string grounded in actual numbers.
    """
    parts = []

    if quiz_component is not None or psych_component is not None or voice_component is not None:
        q_part = (
            f"{round(quiz_component)}% from your AI Career Quiz"
            if quiz_component is not None
            else None
        )
        p_part = (
            f"{round(psych_component)}% from your psychological profile"
            if psych_component is not None
            else None
        )
        v_part = (
            f"{round(voice_component)}% from your voice response"
            if voice_component is not None
            else None
        )
        numeric_bits = [b for b in [q_part, p_part, v_part] if b]
        if numeric_bits:
            parts.append(
                f"This career was recommended with an overall alignment of {round(final_score)}%, "
                f"combining " + ", ".join(numeric_bits) + "."
            )

    if top_traits:
        trait_bits = [
            f"{_prettify_skill_key(t['trait'])} ({round(t['user_score'])}%)"
            for t in top_traits
        ]
        parts.append(
            "Your psychological profile shows strong levels in "
            + ", ".join(trait_bits)
            + f", which are especially important for {career}."
        )

    if quiz_signals:
        ml_prob = quiz_signals.get("ml_probability")
        acad = quiz_signals.get("academic_percent")
        matched_skills = quiz_signals.get("matched_skills") or []
        skill_text = ""
        if matched_skills:
            skill_text = (
                " and key skills such as " + ", ".join(matched_skills)
            )
        if ml_prob is not None and acad is not None:
            parts.append(
                f"The quiz model shows about {round(ml_prob * 100)}% confidence in this path,"
                f" and your academic performance around {round(acad)}%{skill_text} further supports this recommendation."
            )
        elif ml_prob is not None:
            parts.append(
                f"The quiz model shows about {round(ml_prob * 100)}% confidence in this path{skill_text}."
            )
        elif acad is not None and skill_text:
            parts.append(
                f"Your academic performance around {round(acad)}%{skill_text} also supports this recommendation."
            )

    if confidence_score is not None and alt_career:
        if confidence_score < 5.0:
            parts.append(
                f"Your profile also closely matches {alt_career}, with a very similar overall score."
            )
        elif confidence_score < 10.0:
            parts.append(
                f"There is also a moderate alignment with {alt_career}, which could be considered as an adjacent option."
            )

    return " ".join(parts).strip()


# ================== ML Routes ==================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/psych-status", methods=["GET"])
def psych_status():
    """
    Lightweight status endpoint used by the frontend to gate access
    to the AI Career Quiz. Returns whether the user has completed
    at least one psychological assessment.
    """
    user_email = request.args.get("user_email")
    if not user_email:
        return jsonify({"error": "user_email is required"}), 400

    completed = has_completed_psych(user_email)
    return jsonify({"completed": completed}), 200


@app.route("/api/voice-analysis", methods=["POST"])
def voice_analysis():
    """
    Accept audio file, transcribe to text, analyze NLP signals,
    compute voice career scores, store in history, and update fusion.
    """
    payload = None
    if request.is_json:
        payload = request.get_json(silent=True) or {}

    user_email = request.form.get("user_email") or (payload or {}).get("user_email")
    if not user_email:
        return jsonify({"error": "user_email is required"}), 400

    transcript_in = request.form.get("transcript")
    if transcript_in is None and payload:
        transcript_in = payload.get("transcript")

    audio_file = request.files.get("audio")

    # Debug logging (never silent)
    try:
        print(
            "[voice-analysis] user_email=",
            user_email,
            "| has_transcript=",
            bool(transcript_in and str(transcript_in).strip()),
            "| transcript_len=",
            len(str(transcript_in or "")),
            "| has_audio=",
            bool(audio_file and audio_file.filename),
            "| audio_mimetype=",
            getattr(audio_file, "mimetype", None),
            "| audio_filename=",
            getattr(audio_file, "filename", None),
        )
    except Exception:
        pass

    transcribed_text = None
    stt_used = False

    if transcript_in and isinstance(transcript_in, str) and transcript_in.strip():
        transcribed_text = transcript_in.strip()
    else:
        if not audio_file or audio_file.filename == "":
            return jsonify({"error": "Either transcript or audio file is required"}), 400

        audio_bytes = audio_file.read() or b""
        try:
            print(
                "[voice-analysis] audio_size_bytes=",
                len(audio_bytes),
                "| audio_mimetype=",
                getattr(audio_file, "mimetype", None),
            )
        except Exception:
            pass

        if len(audio_bytes) < 5000:
            return jsonify({"error": "Recording too short. Please record at least 5 seconds."}), 400

        format_hint = "webm" if "webm" in (audio_file.filename or "").lower() else "webm"
        try:
            transcribed_text = _transcribe_audio(audio_bytes, format_hint)
            stt_used = True
        except Exception as e:
            print("[voice-analysis] STT exception:", repr(e))
            return (
                jsonify(
                    {
                        "error": "Could not transcribe audio. Please ensure clear speech and try again.",
                        "transcribed_text": "",
                        "transcript_preview": (transcript_in or "").strip() if isinstance(transcript_in, str) else "",
                        "debug": {
                            "stt_exception": type(e).__name__,
                            "audio_size_bytes": len(audio_bytes),
                            "audio_mimetype": getattr(audio_file, "mimetype", None),
                        },
                    }
                ),
                400,
            )

    if not transcribed_text or not str(transcribed_text).strip():
        return (
            jsonify(
                {
                    "error": "Transcript empty. Please try again.",
                    "transcribed_text": "",
                    "transcript_preview": (transcript_in or "").strip() if isinstance(transcript_in, str) else "",
                }
            ),
            400,
        )

    # Minimum word validation (prevents short/empty captures and NaN downstream)
    word_count = len(str(transcribed_text).strip().split())
    try:
        print("[voice-analysis] transcript_word_count=", word_count)
    except Exception:
        pass
    if word_count < 10:
        return (
            jsonify(
                {
                    "error": "Transcript too short. Please speak at least 10 words.",
                    "transcribed_text": str(transcribed_text).strip(),
                }
            ),
            400,
        )

    try:
        analysis = _analyze_voice_text(transcribed_text)
    except Exception as e:
        print("Voice NLP error:", e)
        analysis = {"sentiment": 0.5, "motivation_score": 50.0, "confidence_score": 50.0}

    sentiment = _safe_number(analysis.get("sentiment", 0.5), 0.5)
    confidence = _safe_number(analysis.get("confidence_score", 50.0), 50.0)
    voice_scores = _compute_voice_career_scores(transcribed_text, sentiment, confidence)

    top_voice_career = None
    if voice_scores:
        top_voice_career = max(voice_scores.items(), key=lambda kv: kv[1])[0]

    _save_voice_entry(
        user_email,
        transcript=transcribed_text,
        voice_scores=voice_scores,
        metadata={
            "sentiment": sentiment,
            "motivation_score": analysis.get("motivation_score"),
            "confidence_score": confidence,
        },
    )

    quiz_scores, academic_percent, skills_vector = _get_quiz_context(user_email)
    psych_scores = _get_latest_psych_scores(user_email)
    fusion_result = _fuse_career_scores(
        quiz_scores, psych_scores, academic_percent, skills_vector, voice_scores
    )
    _update_fused_results(
        user_email,
        quiz_scores=quiz_scores,
        psych_scores=psych_scores,
        fusion_result=fusion_result,
        academic_percent=academic_percent,
        skills_vector=skills_vector,
        voice_scores=voice_scores,
    )

    return jsonify({
        "transcribed_text": transcribed_text,
        "sentiment": sentiment,
        "motivation_score": _safe_number(analysis.get("motivation_score"), 50.0),
        "confidence_score": confidence,
        "keyword_alignment": voice_scores,
        "top_voice_career": top_voice_career,
        "voice_scores": voice_scores,
        "stt_used": stt_used,
    }), 200


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

    quiz_scores, academic_percent, skills_vector = _get_quiz_context(user_email)
    voice_scores = _get_latest_voice_scores(user_email)
    fusion_result = _fuse_career_scores(
        quiz_scores, psych_scores, academic_percent, skills_vector, voice_scores
    )
    _update_fused_results(
        user_email,
        quiz_scores=quiz_scores,
        psych_scores=psych_scores,
        fusion_result=fusion_result,
        academic_percent=academic_percent,
        skills_vector=skills_vector,
        voice_scores=voice_scores,
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
                    "id": existing.get("id") or legacy_ts,
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

        attempt_id = str(uuid4())
        now_ts = datetime.utcnow().isoformat() + "Z"
        # Confidence: best quiz score in this attempt (0–100)
        confidence = None
        if isinstance(quiz_scores, dict) and quiz_scores:
            try:
                confidence = max(float(v) for v in quiz_scores.values())
            except (TypeError, ValueError):
                confidence = None

        attempts.insert(
            0,
            {
                "id": attempt_id,
                "timestamp": now_ts,
                "quiz_scores": quiz_scores,
                "academic_percent": academic_percent,
                "stream": stream,
                "top_career": top_career,
                "confidence": confidence,
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

    psych_scores = _get_latest_psych_scores(user_email)
    voice_scores = _get_latest_voice_scores(user_email)
    fusion_result = _fuse_career_scores(
        quiz_scores, psych_scores, academic_percent, skills_vector, voice_scores
    )
    _update_fused_results(
        user_email,
        quiz_scores=quiz_scores,
        psych_scores=psych_scores,
        fusion_result=fusion_result,
        academic_percent=academic_percent,
        skills_vector=skills_vector,
        voice_scores=voice_scores,
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
    psych_profile = _get_latest_psych_profile(user_email)
    voice_scores = _get_latest_voice_scores(user_email)

    fusion_result = _fuse_career_scores(
        quiz_scores, psych_scores, academic_percent, skills_vector, voice_scores
    )
    _update_fused_results(
        user_email,
        quiz_scores=quiz_scores,
        psych_scores=psych_scores,
        fusion_result=fusion_result,
        academic_percent=academic_percent,
        skills_vector=skills_vector,
        voice_scores=voice_scores,
    )

    rankings = fusion_result.get("career_rankings") or []
    top_recommendation = None

    if rankings:
        top = rankings[0]
        career = top.get("career")
        final_score = float(top.get("final_score", 0.0))
        quiz_component = top.get("quiz_component")
        psych_component = top.get("psych_component")
        voice_component = top.get("voice_component")
        quiz_contribution = top.get("quiz_contribution")
        psych_contribution = top.get("psych_contribution")
        voice_contribution = top.get("voice_contribution")

        # Confidence score derived from unified engine
        confidence_score = fusion_result.get("confidence_score")
        alt_career = None
        if len(rankings) > 1:
            second = rankings[1]
            gap = float(top.get("final_score", 0.0)) - float(second.get("final_score", 0.0))
            alt_career = second.get("career")

        top_traits = _top_traits_for_career(psych_profile, career)
        quiz_signals = _quiz_signals_for_career(
            career, quiz_scores, academic_percent, skills_vector
        )

        explanation = _generate_explanation(
            career=career,
            final_score=final_score,
            quiz_component=quiz_component,
            psych_component=psych_component,
            quiz_contribution=quiz_contribution,
            psych_contribution=psych_contribution,
            top_traits=top_traits,
            quiz_signals=quiz_signals,
            confidence_score=confidence_score,
            alt_career=alt_career,
        )

        top_recommendation = {
            "career": career,
            "final_score": final_score,
            "quiz_component": quiz_component,
            "psych_component": psych_component,
            "voice_component": voice_component,
            "quiz_contribution": quiz_contribution,
            "psych_contribution": psych_contribution,
            "voice_contribution": voice_contribution,
            "top_traits": top_traits,
            "quiz_signals": quiz_signals,
            "academic_percent": academic_percent,
            "confidence_score": confidence_score,
            "also_close_career": alt_career,
            "explanation": explanation,
        }

    voice_insight = None
    if voice_scores:
        history = voice_db.get(user_email) or []
        latest = history[0] if history else None
        if latest:
            voice_insight = {
                "transcript": latest.get("transcript", "")[:200] + ("..." if len(latest.get("transcript", "")) > 200 else ""),
                "motivation_score": latest.get("motivation_score"),
                "confidence_score": latest.get("confidence_score"),
                "top_voice_career": max(voice_scores.items(), key=lambda kv: kv[1])[0] if voice_scores else None,
            }

    return jsonify(
        {
            "career_rankings": rankings,
            "quiz_scores": quiz_scores,
            "psych_scores": psych_scores,
            "voice_scores": voice_scores,
            "academic_percent": academic_percent,
            "skills_vector": skills_vector,
            "top_recommendation": top_recommendation,
            "voice_insight": voice_insight,
        }
    )


@app.route("/api/quiz-attempt", methods=["GET"])
def get_quiz_attempt():
    """
    Return full details for a specific AI Career Quiz attempt, including
    per-attempt quiz scores plus a fresh fused ranking and explanation.
    """
    user_email = request.args.get("user_email")
    attempt_id = request.args.get("attempt_id")

    if not user_email:
        return jsonify({"error": "user_email is required"}), 400
    if not attempt_id:
        return jsonify({"error": "attempt_id is required"}), 400

    snapshot = quiz_db.get(user_email) or {}
    attempts = snapshot.get("attempts") or []

    # Support both UUID-based id and legacy timestamp-only identifiers
    target = None
    for a in attempts:
        if a.get("id") == attempt_id or a.get("timestamp") == attempt_id:
            target = a
            break

    if not target:
        return jsonify({"error": "Attempt not found"}), 404

    quiz_scores = target.get("quiz_scores") or {}
    academic_percent = target.get("academic_percent")
    stream = target.get("stream")
    confidence = target.get("confidence")

    psych_scores = _get_latest_psych_scores(user_email)
    psych_profile = _get_latest_psych_profile(user_email)
    voice_scores = _get_latest_voice_scores(user_email)

    fusion_result = _fuse_career_scores(
        quiz_scores, psych_scores, academic_percent, None, voice_scores
    )
    rankings = fusion_result.get("career_rankings") or []

    top_recommendation = None
    if rankings:
        top = rankings[0]
        career = top.get("career")
        final_score = float(top.get("final_score", 0.0))
        quiz_component = top.get("quiz_component")
        psych_component = top.get("psych_component")
        voice_component = top.get("voice_component")
        quiz_contribution = top.get("quiz_contribution")
        psych_contribution = top.get("psych_contribution")
        voice_contribution = top.get("voice_contribution")

        # Confidence index from unified engine
        confidence_score = fusion_result.get("confidence_score")
        alt_career = None
        if len(rankings) > 1:
            second = rankings[1]
            gap = float(top.get("final_score", 0.0)) - float(second.get("final_score", 0.0))
            alt_career = second.get("career")

        top_traits = _top_traits_for_career(psych_profile, career)
        quiz_signals = _quiz_signals_for_career(
            career, quiz_scores, academic_percent, None
        )

        explanation = _generate_explanation(
            career=career,
            final_score=final_score,
            quiz_component=quiz_component,
            psych_component=psych_component,
            quiz_contribution=quiz_contribution,
            psych_contribution=psych_contribution,
            top_traits=top_traits,
            quiz_signals=quiz_signals,
            confidence_score=confidence_score,
            alt_career=alt_career,
        )

        top_recommendation = {
            "career": career,
            "final_score": final_score,
            "quiz_component": quiz_component,
            "psych_component": psych_component,
            "voice_component": voice_component,
            "quiz_contribution": quiz_contribution,
            "psych_contribution": psych_contribution,
            "voice_contribution": voice_contribution,
            "top_traits": top_traits,
            "quiz_signals": quiz_signals,
            "academic_percent": academic_percent,
            "confidence_score": confidence_score,
            "also_close_career": alt_career,
            "explanation": explanation,
        }

    voice_insight = None
    if voice_scores:
        history = voice_db.get(user_email) or []
        latest = history[0] if history else None
        if latest:
            voice_insight = {
                "transcript": latest.get("transcript", "")[:200] + ("..." if len(latest.get("transcript", "")) > 200 else ""),
                "motivation_score": latest.get("motivation_score"),
                "confidence_score": latest.get("confidence_score"),
                "top_voice_career": max(voice_scores.items(), key=lambda kv: kv[1])[0] if voice_scores else None,
            }

    return jsonify(
        {
            "attempt": {
                "id": target.get("id"),
                "timestamp": target.get("timestamp"),
                "quiz_scores": quiz_scores,
                "academic_percent": academic_percent,
                "stream": stream,
                "top_career": target.get("top_career"),
                "confidence": confidence,
            },
            "career_rankings": rankings,
            "top_recommendation": top_recommendation,
            "voice_insight": voice_insight,
        }
    ), 200


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

    # Ensure each attempt has a stable id for frontend navigation
    for attempt in attempts_sorted:
        if not attempt.get("id"):
            # Fallback: use timestamp as identifier if missing
            attempt["id"] = attempt.get("timestamp") or datetime.utcnow().isoformat() + "Z"

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
        quiz_scores, academic_percent, skills_vector = _get_quiz_context(user_email)
        psych_scores = _get_latest_psych_scores(user_email)
        voice_scores = _get_latest_voice_scores(user_email)
        fusion_result = _fuse_career_scores(
            quiz_scores, psych_scores, academic_percent, skills_vector, voice_scores
        )
        _update_fused_results(
            user_email,
            quiz_scores=quiz_scores,
            psych_scores=psych_scores,
            fusion_result=fusion_result,
            academic_percent=academic_percent,
            skills_vector=skills_vector,
            voice_scores=voice_scores,
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
            quiz_scores, academic_percent, skills_vector = _get_quiz_context(user_email)
            psych_scores = _get_latest_psych_scores(user_email)
            voice_scores = _get_latest_voice_scores(user_email)
            fusion_result = _fuse_career_scores(
                quiz_scores, psych_scores, academic_percent, skills_vector, voice_scores
            )
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
