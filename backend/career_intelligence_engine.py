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
    # Clamp to [0, 100] as per score specification
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
        # Normalize by full scale (0–100) to get dimensionless measure
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
        # Fallback: rediscover using raw base weights
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
    # Keep within [0, 1] for downstream interpretation
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

    # Agreement metrics and confidence score
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
        # Normalize margin to [0,1] using full 0–100 scale
        margin_norm = max(0.0, min(1.0, margin / 100.0))
        agree = max(0.0, min(1.0, signal_agreement.get("overall", 0.0)))
        # Combine both aspects symmetrically into a 0–100 confidence index
        confidence_score = 100.0 * (0.5 * margin_norm + 0.5 * agree)

    return {
        "career_rankings": rankings,
        "weights": weights,
        "signal_agreement": signal_agreement,
        "confidence_score": float(_safe_number(confidence_score, 0.0)),
    }

