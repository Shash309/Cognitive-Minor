#!/usr/bin/env python
"""
=============================================================================
  Cognitive-Minor: Career Recommendation — Result Visualizations
=============================================================================
  Generates classification metrics, confusion matrices, feature importance,
  sample predictions, psychometric radar charts, and model comparison graphs.

  Usage:  python generate_visualizations.py
  Output: PNG files saved to  backend/results/
=============================================================================
"""

import os, sys, json, math, random, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR  = BASE_DIR / "data"
OUT_DIR   = BASE_DIR / "results"
OUT_DIR.mkdir(exist_ok=True)

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 0 — Utility helpers
# ════════════════════════════════════════════════════════════════════════════

def _hr(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def _save_fig(fig, name: str):
    path = OUT_DIR / f"{name}.png"
    fig.savefig(str(path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] Saved -> {path}")

def _load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — Load models
# ════════════════════════════════════════════════════════════════════════════

def _try_load(name):
    import joblib
    p = MODEL_DIR / name
    if p.exists():
        obj = joblib.load(str(p))
        print(f"  [OK] Loaded {name}")
        return obj
    print(f"  [--] Missing {name}")
    return None

_hr("Loading Models")
quiz_model    = _try_load("career_1200_model.pkl")
quiz_vec      = _try_load("quiz_vectorizer.pkl")
quiz_enc      = _try_load("quiz_label_encoder.pkl")
psych_model   = _try_load("psych_model.pkl")
psych_enc     = _try_load("psych_label_encoder.pkl")
voice_model   = _try_load("voice_model_v2.pkl")
voice_enc     = _try_load("voice_label_encoder_v2.pkl")

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — Reconstruct psych test data from saved profiles
# ════════════════════════════════════════════════════════════════════════════

OCEAN_KEYS = ["openness", "conscientiousness", "extraversion",
              "agreeableness", "neuroticism"]

psych_X_test = None
psych_y_test = None
psych_y_pred = None

if psych_model and psych_enc:
    _hr("Reconstructing Psych Test Data")
    try:
        profiles_db = _load_json(DATA_DIR / "psych_profiles.json")
        rows_X, rows_y = [], []
        for email, entries in profiles_db.items():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                profile = entry.get("profile", {})
                feats = [float(profile.get(k, 50.0)) for k in OCEAN_KEYS]
                # Use the top career match as ground truth
                matches = entry.get("career_matches", [])
                if matches:
                    label = matches[0].get("career", None)
                    if label:
                        rows_X.append(feats)
                        rows_y.append(label)
        if rows_X:
            psych_X_test = np.array(rows_X)
            psych_y_test = np.array(rows_y)
            psych_y_pred = psych_enc.inverse_transform(
                psych_model.predict(psych_X_test)
            )
            print(f"  Built {len(rows_X)} samples from psych_profiles.json")
        else:
            print("  No usable psych profile entries found.")
    except Exception as e:
        print(f"  [WARN] Psych reconstruction failed: {e}")

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — Classification Metrics  (psych model)
# ════════════════════════════════════════════════════════════════════════════

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)

if psych_y_test is not None and psych_y_pred is not None:
    _hr("Classification Metrics — Psych Model")
    acc  = accuracy_score(psych_y_test, psych_y_pred)
    prec = precision_score(psych_y_test, psych_y_pred, average="weighted", zero_division=0)
    rec  = recall_score(psych_y_test, psych_y_pred, average="weighted", zero_division=0)
    f1   = f1_score(psych_y_test, psych_y_pred, average="weighted", zero_division=0)

    print(f"\n  {'Metric':<15} {'Value':>10}")
    print(f"  {'-'*28}")
    print(f"  {'Accuracy':<15} {acc:>10.4f}")
    print(f"  {'Precision':<15} {prec:>10.4f}")
    print(f"  {'Recall':<15} {rec:>10.4f}")
    print(f"  {'F1-Score':<15} {f1:>10.4f}")
    print(f"\n  Full Classification Report:\n")
    print(classification_report(psych_y_test, psych_y_pred, zero_division=0))
else:
    print("\n  [SKIP] Skipped classification metrics (no test data available).")

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — Confusion Matrix
# ════════════════════════════════════════════════════════════════════════════

if psych_y_test is not None and psych_y_pred is not None:
    _hr("Confusion Matrix — Psych Model")
    try:
        labels = sorted(set(psych_y_test) | set(psych_y_pred))
        cm = confusion_matrix(psych_y_test, psych_y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(max(8, len(labels)*0.9), max(6, len(labels)*0.7)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=True)
        ax.set_title("Confusion Matrix — Psych Career Model", fontsize=14, pad=16)
        fig.tight_layout()
        _save_fig(fig, "confusion_matrix")
    except Exception as e:
        print(f"  [WARN] Failed: {e}")

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — Feature Importance
# ════════════════════════════════════════════════════════════════════════════

def _extract_importances(model, feature_names):
    """Try multiple strategies to get feature importances."""
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    if hasattr(model, "coef_"):
        c = np.abs(model.coef_)
        return np.mean(c, axis=0) if c.ndim > 1 else c
    # VotingClassifier — try sub-estimators
    if hasattr(model, "estimators_"):
        for est in model.estimators_:
            imp = _extract_importances(est, feature_names)
            if imp is not None:
                return imp
    return None

_hr("Feature Importance")

# 5a — Psych model (small, interpretable)
if psych_model:
    imp = _extract_importances(psych_model, OCEAN_KEYS)
    if imp is not None and len(imp) == len(OCEAN_KEYS):
        names = [k.replace("_", " ").title() for k in OCEAN_KEYS]
        idx = np.argsort(imp)[::-1]
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
        ax.barh(range(len(names)), imp[idx][::-1], color=colors[::-1])
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([names[i] for i in idx][::-1])
        ax.set_xlabel("Importance Score")
        ax.set_title("Feature Importance — Psych Model (OCEAN Traits)", fontsize=13, pad=12)
        fig.tight_layout()
        _save_fig(fig, "feature_importance_psych")
    else:
        print("  Psych model: no extractable feature importances.")

# 5b — Quiz model (TF-IDF vocabulary)
if quiz_model and quiz_vec:
    imp = _extract_importances(quiz_model, None)
    if imp is not None:
        try:
            feat_names = quiz_vec.get_feature_names_out()
        except Exception:
            feat_names = [f"tfidf_{i}" for i in range(len(imp))]
        # Only take top-10 from the TF-IDF portion
        n_tfidf = len(feat_names)
        # The model may have embeddings + tfidf; take last n_tfidf
        if len(imp) > n_tfidf:
            tfidf_imp = imp[-n_tfidf:]
        else:
            tfidf_imp = imp[:n_tfidf]
            feat_names = feat_names[:len(imp)]
        top_idx = np.argsort(tfidf_imp)[::-1][:10]
        top_names = [feat_names[i] for i in top_idx]
        top_vals  = tfidf_imp[top_idx]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.magma(np.linspace(0.25, 0.85, 10))
        ax.barh(range(10), top_vals[::-1], color=colors)
        ax.set_yticks(range(10))
        ax.set_yticklabels(top_names[::-1])
        ax.set_xlabel("Importance Score")
        ax.set_title("Top 10 Feature Importances — Quiz Model (TF-IDF Vocab)", fontsize=13, pad=12)
        fig.tight_layout()
        _save_fig(fig, "feature_importance_quiz")
    else:
        print("  Quiz model: no extractable feature importances.")

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — Sample Predictions (Psych)
# ════════════════════════════════════════════════════════════════════════════

if psych_X_test is not None and psych_y_test is not None and psych_y_pred is not None:
    _hr("Sample Predictions — Psych Model")
    n = min(3, len(psych_y_test))
    indices = random.sample(range(len(psych_y_test)), n)
    for i, idx in enumerate(indices, 1):
        print(f"\n  Sample {i} (index {idx}):")
        feats = {OCEAN_KEYS[j]: f"{psych_X_test[idx][j]:.1f}" for j in range(5)}
        print(f"    Input (OCEAN): {feats}")
        print(f"    True Label:      {psych_y_test[idx]}")
        print(f"    Predicted Label: {psych_y_pred[idx]}")
        if hasattr(psych_model, "predict_proba"):
            try:
                proba = psych_model.predict_proba(psych_X_test[idx:idx+1])[0]
                print(f"    Confidence:      {np.max(proba)*100:.2f}%")
            except Exception:
                pass

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — Psychometric Radar Chart (OCEAN from real profiles)
# ════════════════════════════════════════════════════════════════════════════

_hr("Psychometric Visualization — OCEAN Radar")
try:
    profiles_db = _load_json(DATA_DIR / "psych_profiles.json")
    # Collect all profiles and compute average OCEAN
    all_profiles = []
    for email, entries in profiles_db.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            p = entry.get("profile", {})
            if p:
                all_profiles.append(p)

    if all_profiles:
        extended_traits = ["openness", "conscientiousness", "extraversion",
                           "agreeableness", "neuroticism", "analytical_thinking",
                           "risk_tolerance", "leadership_index", "stress_tolerance"]
        trait_labels = [t.replace("_", " ").title() for t in extended_traits]
        avg_vals = []
        for t in extended_traits:
            vals = [float(p.get(t, 50.0)) for p in all_profiles]
            avg_vals.append(np.mean(vals))

        # Radar
        num = len(extended_traits)
        angles = [n / float(num) * 2 * math.pi for n in range(num)]
        angles += angles[:1]
        avg_vals_plot = list(avg_vals) + [avg_vals[0]]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.plot(angles, avg_vals_plot, linewidth=2.5, linestyle="solid", color="#5B21B6", label="Avg Profile")
        ax.fill(angles, avg_vals_plot, color="#5B21B6", alpha=0.18)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(trait_labels, size=9)
        ax.set_ylim(0, 100)
        ax.set_title(f"Avg Psychometric Profile ({len(all_profiles)} assessments)",
                     size=14, color="#5B21B6", pad=24)
        ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1.1))
        fig.tight_layout()
        _save_fig(fig, "ocean_radar")

        # Also plot individual user radars (first 3 unique users)
        users_plotted = 0
        fig2, axes = plt.subplots(1, min(3, len(profiles_db)), figsize=(6*min(3, len(profiles_db)), 6),
                                  subplot_kw=dict(polar=True))
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        for email, entries in profiles_db.items():
            if users_plotted >= 3:
                break
            if not isinstance(entries, list) or not entries:
                continue
            p = entries[-1].get("profile", {})
            if not p:
                continue
            vals = [float(p.get(t, 50.0)) for t in extended_traits] + [float(p.get(extended_traits[0], 50.0))]
            ax2 = axes[users_plotted]
            ax2.plot(angles, vals, linewidth=2, color=plt.cm.Set1(users_plotted))
            ax2.fill(angles, vals, alpha=0.15, color=plt.cm.Set1(users_plotted))
            ax2.set_xticks(angles[:-1])
            ax2.set_xticklabels(trait_labels, size=7)
            ax2.set_ylim(0, 100)
            short_email = email.split("@")[0][:12]
            ax2.set_title(short_email, size=11, pad=14)
            users_plotted += 1
        fig2.suptitle("Individual Psychometric Profiles", fontsize=14, y=1.02)
        fig2.tight_layout()
        _save_fig(fig2, "ocean_radar_individual")
    else:
        print("  No psych profiles found — skipping radar chart.")
except Exception as e:
    print(f"  ⚠ Radar chart failed: {e}")

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — Model Comparison Bar Chart
# ════════════════════════════════════════════════════════════════════════════

_hr("Model Comparison")
try:
    model_names = []
    model_accs  = []

    # Psych accuracy (already computed)
    if psych_y_test is not None and psych_y_pred is not None:
        model_names.append("Psych\n(OCEAN→Career)")
        model_accs.append(accuracy_score(psych_y_test, psych_y_pred))

    # Voice model — quick self-check using psych embeddings as proxy
    if voice_model and voice_enc and psych_X_test is not None:
        try:
            # Voice model expects embeddings; psych features are too small.
            # Just report that it exists with its class count.
            n_classes = len(voice_enc.classes_)
            print(f"  Voice model loaded: {n_classes} career classes — {list(voice_enc.classes_)}")
            model_names.append(f"Voice\n({n_classes} classes)")
            # Cannot compute real accuracy without voice test data — mark N/A
            model_accs.append(0)  # placeholder
        except Exception:
            pass

    # Quiz model class info
    if quiz_model and quiz_enc:
        try:
            n_classes = len(quiz_enc.classes_)
            print(f"  Quiz model loaded: {n_classes} career classes — {list(quiz_enc.classes_)}")
            model_names.append(f"Quiz\n({n_classes} classes)")
            model_accs.append(0)  # no test set available
        except Exception:
            pass

    if model_names:
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#10B981" if a > 0 else "#94A3B8" for a in model_accs]
        bars = ax.bar(model_names, [a * 100 if a > 0 else 0 for a in model_accs], color=colors, width=0.5)
        for bar, acc in zip(bars, model_accs):
            if acc > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f"{acc*100:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=12)
            else:
                ax.text(bar.get_x() + bar.get_width()/2, 2,
                        "No test data", ha="center", va="bottom", fontsize=9, color="#64748B")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 110)
        ax.set_title("Model Comparison — Career Recommendation Pipeline", fontsize=14, pad=14)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        _save_fig(fig, "model_comparison")
    else:
        print("  No models available for comparison.")
except Exception as e:
    print(f"  [WARN] Model comparison failed: {e}")

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — Model Architecture Summary (for research paper)
# ════════════════════════════════════════════════════════════════════════════

_hr("Model Architecture Summary")

def _describe_model(name, model, encoder):
    if not model:
        print(f"  {name}: not loaded")
        return
    print(f"\n  {name}:")
    print(f"    Type:            {type(model).__name__}")
    if hasattr(model, "n_features_in_"):
        print(f"    Input features:  {model.n_features_in_}")
    if encoder is not None:
        classes = list(encoder.classes_) if hasattr(encoder, "classes_") else "N/A"
        print(f"    Output classes:  {len(classes)} -> {classes}")
    if hasattr(model, "estimators_"):
        sub = [type(e).__name__ for e in model.estimators_]
        print(f"    Sub-estimators:  {sub}")
    if hasattr(model, "n_estimators"):
        print(f"    n_estimators:    {model.n_estimators}")

_describe_model("Quiz Model (career_1200_model)", quiz_model, quiz_enc)
_describe_model("Psych Model (psych_model)", psych_model, psych_enc)
_describe_model("Voice Model (voice_model_v2)", voice_model, voice_enc)

# ════════════════════════════════════════════════════════════════════════════
#  DONE
# ════════════════════════════════════════════════════════════════════════════
_hr("COMPLETE")
print(f"\n  All outputs saved to: {OUT_DIR}")
print(f"  Files generated:")
for f in sorted(OUT_DIR.glob("*.png")):
    print(f"    [*] {f.name}")
print()
