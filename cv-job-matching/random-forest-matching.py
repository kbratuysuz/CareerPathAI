import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --- 1. Veri setlerini yükle ---
with open("dataset/cv-dataset.json", "r", encoding="utf-8") as f:
    cvs = json.load(f)

with open("dataset/job-skills-dataset.json", "r", encoding="utf-8") as f:
    jobs = json.load(f)

# --- 2. Özellik çıkarım fonksiyonu ---
def build_features(cv_skills, job_skills_with_scores):
    cv_skills = set([s.lower() for s in cv_skills])
    job_skills = set([j["skill"].lower() for j in job_skills_with_scores])

    intersection = cv_skills & job_skills
    missing = job_skills - cv_skills
    union = cv_skills | job_skills

    return {
        "skill_match_count": len(intersection),
        "skill_match_ratio": len(intersection) / len(job_skills) if job_skills else 0,
        "cv_skill_count": len(cv_skills),
        "job_skill_count": len(job_skills),
        "missing_skill_count": len(missing),
        "jaccard_similarity": len(intersection) / len(union) if union else 0
    }

# --- 3. Tüm (CV, Job) çiftleri için feature oluştur ---
pairs = []
for job in jobs:
    for cv in cvs:
        f = build_features(cv["skills"], job["skills"])
        f["cv_id"] = cv["cv_id"]
        f["job_id"] = job["job_id"]
        f["label"] = 1 if f["skill_match_ratio"] >= 0.2 else 0
        pairs.append(f)

df = pd.DataFrame(pairs)

# --- 4. Model verisi ---
X = df.drop(columns=["cv_id", "job_id", "label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 5. Random Forest pipeline ---
pipe = Pipeline([
    ("scaler", StandardScaler()),  # random forest aslında scaling’e çok ihtiyaç duymaz ama tutarlılık için bırakıyoruz
    ("rf", RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1
    ))
])

pipe.fit(X_train, y_train)
print("🌲 Test Accuracy:", pipe.score(X_test, y_test))

# --- 6. Feature importance ---
rf_model = pipe.named_steps["rf"]
importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n📊 Feature Importance (Random Forest):\n", importance)

# --- 7. Tahmin skorlarını ekle ---
df["match_score"] = pipe.predict_proba(X)[:, 1]

# --- 8. Örnek: Belirli bir ilan için en uygun CV’leri listele ---
job_id = jobs[0]["job_id"]
print(f"\n🔥 Top matches for {job_id}:")
print(
    df[df["job_id"] == job_id][
        ["cv_id", "match_score", "skill_match_count", "missing_skill_count", "skill_match_ratio"]
    ].sort_values(by="match_score", ascending=False).head(10)
)
