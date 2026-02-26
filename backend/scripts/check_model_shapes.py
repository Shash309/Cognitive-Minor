import joblib
import os
import numpy as np

BASE = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE, "career_1200_model.pkl"))
vec = joblib.load(os.path.join(BASE, "quiz_vectorizer.pkl"))

print("model.n_features_in_:", getattr(model, "n_features_in_", None))
try:
    print("model.coef_.shape:", getattr(model, "coef_", None).shape)
except Exception:
    pass

# TF-IDF vocab size
try:
    tfidf_size = len(vec.get_feature_names_out())
except Exception:
    tfidf_size = None
print("tfidf vocab size:", tfidf_size)

if model is not None and tfidf_size is not None:
    print("expected embedding dim (model - tfidf):", model.n_features_in_ - tfidf_size)
