# %% [markdown]
# # Step 1: Import libraries
# 

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# %% [markdown]
# # Step 2: Load dataset

# %%
df = pd.read_csv("career_quiz_dataset_1200.csv")


# %% [markdown]
# # Peek at the data

# %%
print(df.head())


# %% [markdown]
# # Step 3: Split features/labels

# %%
import pandas as pd

df = pd.read_csv("career_quiz_dataset_1200.csv")

print(df.columns)   # show all column names
print(df.head())    # preview first rows


# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Load data
df = pd.read_csv("career_quiz_dataset_1200.csv")

# Combine all quiz answers into a single text feature
feature_cols = [
    'Q1_Favorite_Subjects', 'Q2_Enjoyed_Activities', 'Q3_Strongest_Skills',
    'Q4_Work_Style', 'Q5_Workplace_Preference', 'Q6_Exam_Readiness',
    'Q7_Location_Preference', 'Q8_Career_Values', 'Q9_LongTerm_Goal',
    'Q10_Academic_Background'
]

df["combined_features"] = df[feature_cols].astype(str).agg(" ".join, axis=1)

# Features and target
X = df["combined_features"]
y = df["Recommended_Career"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build pipeline: TF-IDF + Logistic Regression
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
    ("clf", LogisticRegression(max_iter=2000))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


# %% [markdown]
# Logistic Regression

# %%
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, stop_words="english")),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
])


# %% [markdown]
# Random Forest with Label Encoding

# %%
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Encode target
le = LabelEncoder()
y = le.fit_transform(df["Recommended_Career"])

# Encode categorical text features as strings
X = df[feature_cols].astype(str)

# Simple bag-of-words encoding per column (concat all text)
X_combined = X.agg(" ".join, axis=1)

# Vectorize
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_vec = vectorizer.fit_transform(X_combined)

# Train Random Forest
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_vec, y)

# Evaluate
y_pred = model.predict(X_vec)
print("Train accuracy:", accuracy_score(y, y_pred))


# %% [markdown]
# BERT (Best Long-Term Approach)

# %%
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset

# Load data
df["combined_features"] = df[feature_cols].astype(str).agg(" ".join, axis=1)
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["combined_features"], df["Recommended_Career"], test_size=0.2, stratify=df["Recommended_Career"]
)

# Hugging Face dataset
dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})

# Tokenizer & model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# (need to encode labels to integers here before training)



# %%
pip install datasets

# %% [markdown]
# Train Test Split
# 

# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Test accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))


# %% [markdown]
# 2. Cross-Validation (More Reliable)

# %%
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_vec, y, cv=5)
print("Cross-validation accuracy:", scores.mean())


# %%
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer

# %% [markdown]
# STEP 1: Load your dataset

# %%
df = pd.read_csv("career_quiz_dataset_1200.csv")

print("Columns:", df.columns.tolist())
print("Number of samples:", len(df))

# %%
career_clusters = {
    "Doctor": ["Doctor", "Dentist", "Ayurveda Doctor", "Homeopathy Doctor"],
    "Engineer": ["Software Engineer", "Civil Engineer", "Mechanical Engineer",
                 "IT Engineer", "Embedded Systems Engineer"],
    "Technician": ["Technician - Electrical", "Technician - Mechanical", "IT Support/Technician"],
    "Analyst": ["Business Analyst", "Economist/Analyst", "Investment Analyst", 
                "Financial Analyst", "Public Policy Analyst"],
    "Designer": ["Artist/Designer", "Junior Designer"],
    "Research": ["Researcher/Archivist"],
    "Account/Finance": ["Accountant"],
    "Counseling": ["Counselor/HR"],
}

# Reverse mapping: career → cluster
career_to_cluster = {}
for cluster, careers in career_clusters.items():
    for c in careers:
        career_to_cluster[c] = cluster

# Map to clusters
df["CareerCluster"] = df["Recommended_Career"].map(career_to_cluster)

# Drop rows with unmapped careers
df = df.dropna(subset=["CareerCluster"])

print("Unique career clusters:", df["CareerCluster"].unique())

# %%
# ==============================
# STEP 3: Encode target labels
# ==============================
le = LabelEncoder()
y = le.fit_transform(df["CareerCluster"])


# %%
print("Final dataframe columns:", df.columns.tolist())


# %%
print(df.columns)


# %%
# ==============================
# STEP 4: Create text features
# ==============================

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

# Combine all text columns into one string per student
df_features = df[feature_cols].astype(str)
X_text = df_features.agg(" ".join, axis=1)

# Generate embeddings using sentence-transformers
from sentence_transformers import SentenceTransformer

model_emb = SentenceTransformer("all-MiniLM-L6-v2")
X_vec = model_emb.encode(X_text.tolist(), show_progress_bar=True)


# %%
# ==============================
# STEP 4: Create text features
# ==============================
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

# Combine all text columns into a single string per student
df_features = df[feature_cols].astype(str)
X_text = df_features.agg(" ".join, axis=1)

# Generate embeddings
from sentence_transformers import SentenceTransformer
model_emb = SentenceTransformer("all-MiniLM-L6-v2")
X_vec = model_emb.encode(X_text.tolist(), show_progress_bar=True)

# ==============================
# STEP 5: Train-Test Split
# ==============================
from sklearn.model_selection import train_test_split

y = df["CareerCluster"].values  # target variable
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# STEP 6: Train RandomForest
# ==============================
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",  # handle imbalanced clusters
    random_state=42
)
clf.fit(X_train, y_train)

# ==============================
# STEP 7: Evaluate
# ==============================
from sklearn.metrics import accuracy_score, classification_report

y_pred = clf.predict(X_test)
print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# %%
c

# %%
# ==============================
# STEP 1: Imports
# ==============================
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

# ==============================
# STEP 2: Load dataset
# ==============================
df = pd.read_csv("career_quiz_dataset_1200.csv")  # replace with your CSV path

# ==============================
# STEP 3: Define career clusters
# ==============================
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

# ==============================
# STEP 4: Combine text columns
# ==============================
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

# ==============================
# STEP 5: Create embeddings and TF-IDF features
# ==============================
# Sentence embeddings
model_emb = SentenceTransformer("all-MiniLM-L6-v2")
X_embeddings = model_emb.encode(X_text.tolist(), show_progress_bar=True)

# TF-IDF features
vectorizer = TfidfVectorizer(max_features=3000, stop_words="english")
X_tfidf = vectorizer.fit_transform(X_text)

# Combine embeddings + TF-IDF
X_combined = np.hstack([X_embeddings, X_tfidf.toarray()])

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(df['CareerCluster'])

# ==============================
# STEP 6: Train-Test Split & SMOTE
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# ==============================
# STEP 7: Train Soft Voting Ensemble
# ==============================
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

# ==============================
# STEP 8: Evaluate Model
# ==============================
y_pred = voting_clf.predict(X_test)

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))


# %%
# Run in your notebook after training finishes
import os, joblib
# create folder
os.makedirs("models", exist_ok=True)

# 1) save sentence-transformer (preferred method)
model_emb.save("models/emb_model")   # SentenceTransformer.save

# 2) save TF-IDF vectorizer, classifier and label encoder
joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")
joblib.dump(voting_clf, "models/voting_clf.joblib")
joblib.dump(le, "models/label_encoder.joblib")

# %%
pip install model_emb

# %%
pip install joblib

# %%
import joblib

# Save the trained model
joblib.dump(voting_clf, "career_1200_model.pkl")

# Save the vectorizer (TF-IDF or embeddings)
joblib.dump(vectorizer, "quiz_vectorizer.pkl")

# Save the label encoder
joblib.dump(le, "quiz_label_encoder.pkl")



