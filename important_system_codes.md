# Important System Codes (Cleaned & Minimized)

## Career Intelligence Engine (Fusion Logic)

```python
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
NEUTRAL_SCORE = 50.0
BASE_WEIGHTS = {
    "quiz": 0.4,
    "psych": 0.35,
    "voice": 0.25,
}
def _safe_number(value, default: float = NEUTRAL_SCORE) -> float:
    """
    Robust numeric casting used inside the engine. Always returns a finite float.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(v) or math.isinf(v):
        return default
    if v < 0.0:
        return 0.0
    if v > 100.0:
        return 100.0
    return v
def _standardize_vectors(
    quiz_scores: Optional[Dict[str, float]],
    psych_scores: Optional[Dict[str, float]],
    voice_scores: Optional[Dict[str, float]],
) -> Tuple[List[str], List[float], List[float], List[float]]:
    """
    Ensure all three signals share the same career keys and contain safe scores.
    - Union of all keys becomes the canonical career set.
    - Missing or invalid values are replaced with the neutral 50.
    - All values are clamped to [0, 100].
    """
    quiz_scores = quiz_scores or {}
    psych_scores = psych_scores or {}
    voice_scores = voice_scores or {}
    careers = sorted(
        set(quiz_scores.keys()) | set(psych_scores.keys()) | set(voice_scores.keys())
    )
    q_vec: List[float] = []
    p_vec: List[float] = []
    v_vec: List[float] = []
    for c in careers:
        q_raw = quiz_scores.get(c, NEUTRAL_SCORE)
        p_raw = psych_scores.get(c, NEUTRAL_SCORE)
        v_raw = voice_scores.get(c, NEUTRAL_SCORE)
        q_vec.append(_safe_number(q_raw, NEUTRAL_SCORE))
        p_vec.append(_safe_number(p_raw, NEUTRAL_SCORE))
        v_vec.append(_safe_number(v_raw, NEUTRAL_SCORE))
    return careers, q_vec, p_vec, v_vec
def _adaptive_weights(
    have_quiz: bool,
    have_psych: bool,
    have_voice: bool,
    q_vec: List[float],
    p_vec: List[float],
    v_vec: List[float],
) -> Dict[str, float]:
    """
    Compute adaptive weights for quiz / psych / voice.
    Steps:
    1. Start from base weights (0.4 / 0.35 / 0.25).
    2. Drop any signal that is structurally missing (no scores provided).
    3. Adjust each remaining weight by a strength factor based on score variance:
       - strength = std / 50, clamped to [0, 1] (scale-aware).
       - adjusted_weight = base_weight * (0.5 + 0.5 * strength)
         -> totally flat signal halves its weight; highly varied keeps full weight.
    4. Renormalize so all active weights sum to 1.
    """
    base_q = BASE_WEIGHTS["quiz"] if have_quiz else 0.0
    base_p = BASE_WEIGHTS["psych"] if have_psych else 0.0
    base_v = BASE_WEIGHTS["voice"] if have_voice else 0.0
    if not (base_q or base_p or base_v):
        return {"quiz": 0.0, "psych": 0.0, "voice": 0.0}
    def _strength(values: List[float]) -> float:
        if not values:
            return 0.0
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return 0.0
        std = float(np.std(arr))
        s = std / 50.0
        if s < 0.0:
            s = 0.0
        if s > 1.0:
            s = 1.0
        return s
    s_q = _strength(q_vec) if have_quiz else 0.0
    s_p = _strength(p_vec) if have_psych else 0.0
    s_v = _strength(v_vec) if have_voice else 0.0
    w_q = base_q * (0.5 + 0.5 * s_q)
    w_p = base_p * (0.5 + 0.5 * s_p)
    w_v = base_v * (0.5 + 0.5 * s_v)
    total = w_q + w_p + w_v
    if total <= 0.0:
        total = base_q + base_p + base_v or 1.0
        w_q, w_p, w_v = base_q / total, base_p / total, base_v / total
    else:
        w_q, w_p, w_v = w_q / total, w_p / total, w_v / total
    return {
        "quiz": float(w_q),
        "psych": float(w_p),
        "voice": float(w_v),
    }
def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity in [0, 1] for non-negative vectors.
    Returns 0 when one vector is effectively zero-length.
    """
    if a.size == 0 or b.size == 0:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na <= 1e-8 or nb <= 1e-8:
        return 0.0
    sim = float(np.dot(a, b) / (na * nb))
    if sim < 0.0:
        sim = 0.0
    if sim > 1.0:
        sim = 1.0
    return sim
def _agreement_metrics(
    have_quiz: bool,
    have_psych: bool,
    have_voice: bool,
    q_vec: List[float],
    p_vec: List[float],
    v_vec: List[float],
) -> Dict[str, float]:
    """
    Compute pairwise cosine similarity between signals and an overall agreement score.
    """
    sims = {}
    arr_q = np.asarray(q_vec, dtype=float)
    arr_p = np.asarray(p_vec, dtype=float)
    arr_v = np.asarray(v_vec, dtype=float)
    values: List[float] = []
    if have_quiz and have_psych:
        s = _cosine_similarity(arr_q, arr_p)
        sims["quiz_psych"] = s
        values.append(s)
    else:
        sims["quiz_psych"] = 0.0
    if have_quiz and have_voice:
        s = _cosine_similarity(arr_q, arr_v)
        sims["quiz_voice"] = s
        values.append(s)
    else:
        sims["quiz_voice"] = 0.0
    if have_psych and have_voice:
        s = _cosine_similarity(arr_p, arr_v)
        sims["psych_voice"] = s
        values.append(s)
    else:
        sims["psych_voice"] = 0.0
    overall = float(sum(values) / len(values)) if values else 0.0
    sims["overall"] = overall
    return sims
def compute_final_decision(
    quiz_scores: Optional[Dict[str, float]],
    psych_scores: Optional[Dict[str, float]],
    voice_scores: Optional[Dict[str, float]] = None,
) -> Dict:
    """
    Unified career intelligence engine.
    Inputs:
        quiz_scores: dict[career] -> 0–100
        psych_scores: dict[career] -> 0–100
        voice_scores: optional dict[career] -> 0–100
    Behaviour:
      - Standardizes all signals onto the same career space.
      - Normalizes scores safely to [0, 100] with neutral 50 defaults.
      - Applies adaptive weighting based on availability and variance.
      - Returns:
          {
            "career_rankings": [...],
            "weights": {"quiz": wq, "psych": wp, "voice": wv},
            "signal_agreement": {...},
            "confidence_score": 0–100,
          }
    """
    have_quiz = bool(quiz_scores)
    have_psych = bool(psych_scores)
    have_voice = bool(voice_scores)
    if not have_quiz and not have_psych and not have_voice:
        return {
            "career_rankings": [],
            "weights": {"quiz": 0.0, "psych": 0.0, "voice": 0.0},
            "signal_agreement": {
                "quiz_psych": 0.0,
                "quiz_voice": 0.0,
                "psych_voice": 0.0,
                "overall": 0.0,
            },
            "confidence_score": 0.0,
        }
    careers, q_vec, p_vec, v_vec = _standardize_vectors(
        quiz_scores, psych_scores, voice_scores
    )
    weights = _adaptive_weights(
        have_quiz=have_quiz,
        have_psych=have_psych,
        have_voice=have_voice,
        q_vec=q_vec,
        p_vec=p_vec,
        v_vec=v_vec,
    )
    w_q = weights["quiz"]
    w_p = weights["psych"]
    w_v = weights["voice"]
    rankings = []
    for idx, career in enumerate(careers):
        q = q_vec[idx]
        p = p_vec[idx]
        v = v_vec[idx]
        final_score = (
            (w_q * q if have_quiz else 0.0)
            + (w_p * p if have_psych else 0.0)
            + (w_v * v if have_voice else 0.0)
        )
        final_score = _safe_number(final_score, 0.0)
        quiz_contrib = _safe_number(w_q * q if have_quiz else 0.0, 0.0)
        psych_contrib = _safe_number(w_p * p if have_psych else 0.0, 0.0)
        voice_contrib = _safe_number(w_v * v if have_voice else 0.0, 0.0)
        item = {
            "career": career,
            "final_score": float(final_score),
            "quiz_component": float(q) if have_quiz else None,
            "psych_component": float(p) if have_psych else None,
            "quiz_contribution": float(quiz_contrib),
            "psych_contribution": float(psych_contrib),
        }
        if have_voice:
            item["voice_component"] = float(v)
            item["voice_contribution"] = float(voice_contrib)
        rankings.append(item)
    rankings.sort(key=lambda x: x["final_score"], reverse=True)
    signal_agreement = _agreement_metrics(
        have_quiz=have_quiz,
        have_psych=have_psych,
        have_voice=have_voice,
        q_vec=q_vec,
        p_vec=p_vec,
        v_vec=v_vec,
    )
    confidence_score = 0.0
    if len(rankings) >= 2:
        top1 = _safe_number(rankings[0]["final_score"], 0.0)
        top2 = _safe_number(rankings[1]["final_score"], 0.0)
        margin = max(0.0, top1 - top2)
        margin_norm = max(0.0, min(1.0, margin / 100.0))
        agree = max(0.0, min(1.0, signal_agreement.get("overall", 0.0)))
        confidence_score = 100.0 * (0.5 * margin_norm + 0.5 * agree)
    return {
        "career_rankings": rankings,
        "weights": weights,
        "signal_agreement": signal_agreement,
        "confidence_score": float(_safe_number(confidence_score, 0.0)),
    }
```

## Career Quiz Engine (Adaptive Logic)

```python
import math
def get_static_questions():
    """
    Returns the static questions for Phase 1 of the AI Career Quiz.
    """
    return [
        {
            "id": "Q1",
            "type": "scale",
            "question": "Rate your experience (1-5):",
            "options": [
                "Programming / Coding",
                "Mathematics / Logical Problem Solving",
                "Writing / Communication",
                "Design (UI/UX, creative tools)",
                "Business / Finance",
                "Science (Physics/Chem/Bio)"
            ],
            "scale_range": [1, 5]
        },
        {
            "id": "Q2",
            "type": "multi",
            "question": "What have you actually done?",
            "options": [
                "Built projects",
                "Participated in competitions",
                "Led a team",
                "Internships / real-world work",
                "None"
            ]
        },
        {
            "id": "Q3",
            "type": "multi",
            "question": "Which problems do you enjoy most? (Pick max 2)",
            "options": [
                "Analytical",
                "Creative",
                "Human-centric",
                "Strategic",
                "Scientific"
            ]
        },
        {
            "id": "Q4",
            "type": "single",
            "question": "Work preference:",
            "options": [
                "Build things",
                "Analyze deeply",
                "Manage people",
                "Research/explore"
            ]
        },
        {
            "id": "Q7",
            "type": "single",
            "question": "Uncertainty tolerance:",
            "options": [
                "High",
                "Medium",
                "Low"
            ]
        }
    ]
def _parse_context(context: dict):
    phase1 = context.get("phase1_answers") or {}
    voice = context.get("voice_analysis") or {}
    psych = context.get("psychological_traits") or {}
    def safe_parse(d, k1, k2=None, default=0.5):
        val = d.get(k1)
        if val is None and k2: val = d.get(k2)
        if val is None: return default
        try:
            val_str = str(val).replace('%', '').strip()
            f = float(val_str)
            return f / 100.0 if f > 1.0 else f
        except (ValueError, TypeError):
            return default
    v_conf = safe_parse(voice, "confidence", "confidence_score", 0.5)
    v_hes  = safe_parse(voice, "hesitation", None, 0.5)
    p_open = safe_parse(psych, "openness", None, 0.5)
    p_cons = safe_parse(psych, "conscientiousness", None, 0.5)
    p_ext  = safe_parse(psych, "extraversion", None, 0.5)
    p_agr  = safe_parse(psych, "agreeableness", None, 0.5)
    p_neu  = safe_parse(psych, "neuroticism", None, 0.5)
    base_logical = 0.5
    base_creative = 0.5
    base_social = 0.5
    base_risk = 0.5
    q1 = phase1.get("Q1", {})
    if isinstance(q1, dict):
        base_logical += (q1.get("Programming / Coding", 1) + q1.get("Mathematics / Logical Problem Solving", 1)) * 0.05
        base_creative += q1.get("Design (UI/UX, creative tools)", 1) * 0.1
        base_social += q1.get("Writing / Communication", 1) * 0.1
    q3 = phase1.get("Q3", [])
    if isinstance(q3, list):
        if "Analytical" in q3: base_logical += 0.2
        if "Creative" in q3: base_creative += 0.2
        if "Human-centric" in q3: base_social += 0.2
    q4 = phase1.get("Q4")
    if q4 == "Build things": base_logical += 0.1
    if q4 == "Manage people": base_social += 0.2
    q7 = phase1.get("Q7")
    if q7 == "High": base_risk += 0.3
    if q7 == "Low": base_risk -= 0.2
    return {
        "logical": min(1.0, base_logical),
        "creative": min(1.0, base_creative),
        "social": min(1.0, base_social),
        "risk": min(1.0, max(0.0, base_risk)),
        "voice": {"conf": v_conf, "hes": v_hes},
        "psych": {"open": p_open, "cons": p_cons, "ext": p_ext, "neu": p_neu}
    }
def get_adaptive_questions(context: dict) -> list:
    """
    Phase 2: Adaptive Question Generation
    Rules: Generate 5-10 adaptive questions based on Phase 1, Voice, Psych
    """
    parsed = _parse_context(context)
    questions = []
    idx = 101
    if parsed["logical"] > 0.6:
        questions.extend([
            {"id": f"A_{idx}", "type": "single", "question": "In a technical project, do you prefer:", "options": ["Building from scratch", "Debugging and optimizing", "Designing the system architecture"]},
            {"id": f"A_{idx+1}", "type": "single", "question": "When solving a complex problem, you:", "options": ["Break it down systematically", "Look for an existing pattern", "Experiment until it works"]}
        ])
        idx += 2
    if parsed["creative"] > 0.6 or parsed["psych"]["open"] > 0.7:
        questions.extend([
            {"id": f"A_{idx}", "type": "single", "question": "How often do you generate new concepts?", "options": ["Constantly", "When required", "Rarely"]},
            {"id": f"A_{idx+1}", "type": "single", "question": "Would you rather:", "options": ["Innovate completely new ideas", "Optimize existing structures"]}
        ])
        idx += 2
    if parsed["psych"]["ext"] > 0.6 or parsed["social"] > 0.6:
        questions.extend([
            {"id": f"A_{idx}", "type": "single", "question": "In a team, do you prefer to:", "options": ["Lead and delegate", "Collaborate equally", "Work as an individual contributor"]},
            {"id": f"A_{idx+1}", "type": "single", "question": "Does social interaction:", "options": ["Energize you", "Neutral", "Drain you"]}
        ])
        idx += 2
    elif parsed["psych"]["ext"] < 0.4:
        questions.extend([
            {"id": f"A_{idx}", "type": "single", "question": "For deep work, you prefer:", "options": ["Total isolation", "A quiet shared space", "Headphones in a cafe"]},
            {"id": f"A_{idx+1}", "type": "single", "question": "When faced with a blocker, you:", "options": ["Solve it independently", "Ask for help immediately", "Research online before asking"]}
        ])
        idx += 2
    v_conf = parsed["voice"]["conf"]
    if v_conf < 0.4:
        questions.extend([
            {"id": f"A_{idx}", "type": "single", "question": "When making major decisions, what holds you back?", "options": ["Fear of failure", "Lack of information", "Overthinking"]},
            {"id": f"A_{idx+1}", "type": "single", "question": "How do you handle self-doubt?", "options": ["Talk it out", "Push through it", "Step back and reassess"]}
        ])
        idx += 2
    elif v_conf > 0.7:
        questions.extend([
            {"id": f"A_{idx}", "type": "single", "question": "Are you comfortable taking calculated risks?", "options": ["Absolutely", "If the data supports it", "Only forced to"]},
            {"id": f"A_{idx+1}", "type": "single", "question": "How do you lead?", "options": ["By example", "By strategic delegation", "I prefer being an independent ace"]}
        ])
        idx += 2
    if parsed["risk"] < 0.4 or parsed["psych"]["cons"] > 0.7:
        questions.extend([
            {"id": f"A_{idx}", "type": "single", "question": "You value:", "options": ["Structured environments", "Flexible chaos"]},
            {"id": f"A_{idx+1}", "type": "single", "question": "When planning your career, you look:", "options": ["10 years ahead", "2-3 years ahead", "I improvise"]}
        ])
        idx += 2
    elif parsed["risk"] > 0.6:
        questions.extend([
            {"id": f"A_{idx}", "type": "single", "question": "Startups vs Corporations:", "options": ["High ownership startup", "Stable corporation"]},
            {"id": f"A_{idx+1}", "type": "single", "question": "Are you driven by:", "options": ["High upside/equity", "Predictable growth"]}
        ])
        idx += 2
    return questions[:10]
def extract_core_scores(context: dict, adaptive_answers: dict) -> dict:
    parsed = _parse_context(context)
    logical = parsed["logical"]
    creativity = parsed["creative"]
    social = parsed["social"]
    risk = parsed["risk"]
    leadership = 0.5
    stability = 0.5
    leadership += (parsed["voice"]["conf"] - 0.5) * 0.4
    risk += (parsed["voice"]["conf"] - 0.5) * 0.2
    leadership -= parsed["voice"]["hes"] * 0.2
    creativity += (parsed["psych"]["open"] - 0.5) * 0.3
    stability += (parsed["psych"]["cons"] - 0.5) * 0.4
    for qid, ans in adaptive_answers.items():
        if isinstance(ans, list) and len(ans) > 0: ans = ans[0]
        if not ans: continue
        if ans in ["Designing the system architecture", "System building"]: logical += 0.1; leadership += 0.1
        if ans in ["Innovate completely new ideas", "Constantly"]: creativity += 0.2; risk += 0.1
        if ans in ["Lead and delegate", "High ownership startup"]: leadership += 0.2; risk += 0.2
        if ans in ["Total isolation", "Solve it independently"]: social -= 0.15; logical += 0.1
        if ans in ["Fear of failure", "Overthinking", "Structured environments", "Stable corporation"]: risk -= 0.2; stability += 0.2
        if ans in ["Absolutely", "High upside/equity"]: risk += 0.2; stability -= 0.1
    return {
        "logical": min(1.0, max(0.0, logical)),
        "creativity": min(1.0, max(0.0, creativity)),
        "social": min(1.0, max(0.0, social)),
        "leadership": min(1.0, max(0.0, leadership)),
        "risk": min(1.0, max(0.0, risk)),
        "stability": min(1.0, max(0.0, stability))
    }
def match_careers(scores: dict) -> list:
    clusters = {
        "Software Engineer": {"logical": 0.9, "creativity": 0.4, "social": 0.2, "leadership": 0.3, "risk": 0.4, "stability": 0.7},
        "Product Manager":   {"logical": 0.6, "creativity": 0.6, "social": 0.9, "leadership": 0.9, "risk": 0.6, "stability": 0.5},
        "Designer":          {"logical": 0.3, "creativity": 0.9, "social": 0.6, "leadership": 0.4, "risk": 0.5, "stability": 0.5},
        "Entrepreneur":      {"logical": 0.5, "creativity": 0.8, "social": 0.8, "leadership": 0.9, "risk": 0.9, "stability": 0.1},
        "Data Scientist":    {"logical": 0.9, "creativity": 0.5, "social": 0.3, "leadership": 0.4, "risk": 0.4, "stability": 0.6},
        "Consultant":        {"logical": 0.7, "creativity": 0.5, "social": 0.9, "leadership": 0.7, "risk": 0.6, "stability": 0.4},
        "Researcher":        {"logical": 0.9, "creativity": 0.6, "social": 0.2, "leadership": 0.3, "risk": 0.2, "stability": 0.8}
    }
    results = []
    for career, reqs in clusters.items():
        dist = 0
        for k in scores.keys():
            dist += abs(scores[k] - reqs.get(k, 0.5)) ** 2
        prob = max(0.0, 1.0 - math.sqrt(dist / 6.0))
        results.append({"career": career, "probability": prob})
    results.sort(key=lambda x: x["probability"], reverse=True)
    return results
def process_career_recommendation(context: dict, adaptive_answers: dict) -> dict:
    """
    Final Output Generation matches strict formatting.
    """
    scores = extract_core_scores(context, adaptive_answers)
    career_list = match_careers(scores)
    top = career_list[0]
    reasoning = f"Based on your high alignment with {top['career']} ({round(top['probability']*100)}%), this fits your profile."
    strengths = []
    weaknesses = []
    if scores["logical"] > 0.7: strengths.append("Strong analytical and problem-solving pipeline.")
    if scores["creativity"] > 0.7: strengths.append("High divergent thinking and design skills.")
    if scores["leadership"] > 0.7: strengths.append("Natural leadership and clear delegation.")
    if scores["social"] < 0.4: weaknesses.append("May struggle in highly extroverted cross-functional roles.")
    if scores["risk"] < 0.3: weaknesses.append("Can be overly cautious in rapid-change environments.")
    if not strengths: strengths = ["Balanced versatile skill set."]
    if not weaknesses: weaknesses = ["Generalist profile may lack deep specialization."]
    p_sum = "A balanced analytical mind" if scores["logical"] > scores["creativity"] else "A highly creative and flexible thinker"
    if scores["risk"] > 0.7: p_sum += " who thrives on risk and entrepreneurship."
    elif scores["stability"] > 0.7: p_sum += " who values structured, long-term impact."
    else: p_sum += " with a versatile approach to problem solving."
    v_conf = context.get("voice_analysis", {}).get("confidence", 0.5)
    if v_conf > 0.7:
        c_an = "Voice analysis indicates high confidence and clarity, reinforcing leadership potentials."
    elif v_conf < 0.4:
        c_an = "Voice patterns suggest some hesitation; leaning towards roles providing psychological safety and structure."
    else:
        c_an = "Voice metrics reflect a balanced and measured communication style."
    return {
        "scores": scores,
        "career_recommendations": career_list,
        "top_careers": [{"name": c["career"], "score": c["probability"] * 100} for c in career_list[:3]],
        "insights": {
            "strengths": strengths,
            "weaknesses": weaknesses,
            "personality_summary": p_sum,
            "confidence_analysis": c_an
        },
        "confidence_level": "HIGH" if top["probability"] > 0.8 else "MEDIUM" if top["probability"] > 0.6 else "LOW",
        "reasoning": [reasoning, p_sum, c_an]
    }
```

## Model Training Script (Final Ensemble Pipeline)

```python
import pandas as pd
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
df = pd.read_csv("career_quiz_dataset_1200.csv")  # replace with your CSV path
career_clusters = {
    'Engineering': ['Engineer', 'Technician', 'Embedded Systems Engineer', 'IT Engineer', 'IT Support/Technician', 'Mechanical Engineer', 'Civil Engineer'],
    'Business & Finance': ['Account/Finance', 'Analyst', 'Financial Analyst', 'Investment Analyst', 'Business Analyst'],
    'Design & Creative': ['Designer', 'Artist/Designer', 'UX Designer', 'Graphic Designer', 'Junior Designer'],
    'Healthcare': ['Doctor', 'Counseling', 'Ayurveda Doctor', 'Homeopathy Doctor', 'Dentist'],
    'Research & Academics': ['Researcher', 'Public Policy Analyst', 'Economist/Analyst', 'Researcher/Archivist']
}
def map_to_cluster(career):
    for cluster, careers in career_clusters.items():
        for c in careers:
            if c.lower() in str(career).lower():
                return cluster
    return None
df['CareerCluster'] = df['Recommended_Career'].apply(map_to_cluster)
df = df.dropna(subset=['CareerCluster'])
feature_cols = [
    "Q1_Favorite_Subjects",
    "Q2_Enjoyed_Activities",
    "Q3_Strongest_Skills",
    "Q4_Work_Style",
    "Q5_Workplace_Preference",
    "Q6_Exam_Readiness",
    "Q7_Location_Preference",
    "Q8_Career_Values",
    "Q9_LongTerm_Goal",
    "Q10_Academic_Background"
]
df_text = df[feature_cols].astype(str)
X_text = df_text.agg(" ".join, axis=1)
model_emb = SentenceTransformer("all-MiniLM-L6-v2")
X_embeddings = model_emb.encode(X_text.tolist(), show_progress_bar=True)
vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
X_tfidf = vectorizer.fit_transform(X_text)
X_combined = np.hstack([X_embeddings, X_tfidf.toarray()])
le = LabelEncoder()
y = le.fit_transform(df['CareerCluster'])
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
xgb_clf = XGBClassifier(
    n_estimators=250,
    max_depth=8,
    learning_rate=0.1,
    objective='multi:softprob',  # softprob for probability outputs
    eval_metric='mlogloss',
    random_state=42
)
rf_clf = RandomForestClassifier(
    n_estimators=250,
    max_depth=None,
    class_weight='balanced',
    random_state=42
)
lr_clf = LogisticRegression(
    max_iter=2000,
    class_weight='balanced',
    random_state=42
)
voting_clf = VotingClassifier(
    estimators=[('xgb', xgb_clf), ('rf', rf_clf), ('lr', lr_clf)],
    voting='soft'  # use probabilities to reduce bias toward large classes
)
voting_clf.fit(X_train_res, y_train_res)
y_pred = voting_clf.predict(X_test)
print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
import os, joblib
os.makedirs("models", exist_ok=True)
model_emb.save("models/emb_model")   # SentenceTransformer.save
joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")
joblib.dump(voting_clf, "models/voting_clf.joblib")
joblib.dump(le, "models/label_encoder.joblib")
import joblib
joblib.dump(voting_clf, "career_1200_model.pkl")
joblib.dump(vectorizer, "quiz_vectorizer.pkl")
joblib.dump(le, "quiz_label_encoder.pkl")
```

