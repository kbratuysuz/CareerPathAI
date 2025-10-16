import json
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np


with open("../dataset/resumes/cv-dataset-all.json", "r", encoding="utf-8") as f:cvs = json.load(f)

with open("../dataset/job-postings/job-skills-all.json", "r", encoding="utf-8") as f:jobs = json.load(f)

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

pairs = []
for job in jobs:
    for cv in cvs:
        f = build_features(cv["skills"], job["skills"])
        f["cv_id"] = cv["cv_id"]
        f["job_id"] = job["job_id"]
        f["label"] = 1 if f["skill_match_ratio"] >= 0.2 else 0
        pairs.append(f)

df = pd.DataFrame(pairs)

X = df.drop(columns=["cv_id", "job_id", "label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=500))
])
pipe.fit(X_train, y_train)

print("Test Accuracy:", pipe.score(X_test, y_test))

importance = pd.Series(pipe.named_steps["lr"].coef_[0], index=X.columns).sort_values(ascending=False)
print("\nFeature Importance:\n", importance)

df["match_score"] = pipe.predict_proba(X)[:, 1]

job_id = jobs[0]["job_id"]
print(f"\nTop matches for {job_id}:")
print(
    df[df["job_id"] == job_id][
        ["cv_id", "match_score", "skill_match_count", "missing_skill_count", "skill_match_ratio"]
    ].sort_values(by="match_score", ascending=False).head(10)
)

def simulate_score(cv, job):
    """Bir CV ve job için model skorunu hesapla"""
    features = build_features(cv["skills"], job["skills"])
    X = pd.DataFrame([features])
    return pipe.predict_proba(X)[0][1]

job = jobs[0]
cv = cvs[20]  

job_skills = {j["skill"].lower(): j["score"] for j in job["skills"]}
cv_skills = [s.lower() for s in cv["skills"]]

missing_skills = {s: job_skills[s] for s in job_skills if s not in cv_skills}

base_score = simulate_score(cv, job)

full_cv_skills = list(cv["skills"]) + list(missing_skills.keys())
cv_full = cv.copy()
cv_full["skills"] = full_cv_skills
full_score = simulate_score(cv_full, job)

total_effect = max(full_score - base_score, 0)
if total_effect == 0:
    print(f"\nUygunluk Skoru: {base_score:.2f} (Eksik yetkinlik yok veya etkisiz)")
else:
    total_missing_score = sum(missing_skills.values()) or 1e-6
    weighted_results = []
    for skill, score in missing_skills.items():
        weight = score / total_missing_score
        weighted_increase = weight * total_effect
        weighted_results.append({
            "skill": skill,
            "skill_score": score,
            "weight": weight,
            "score_increase": weighted_increase
        })

    weighted_df = pd.DataFrame(weighted_results).sort_values(by="score_increase", ascending=False)

    print(f"\nUygunluk Skoru: {base_score:.2f}")
    print(f"Tüm Eksik Yetkinlikler Eklense Tahmini Skor: {full_score:.2f}")
    print("Eksik Yetkinlik Önerileri (iş ilanı skorlarına göre ağırlıklı):")

    for _, row in weighted_df.iterrows():
        print(f"  - {row['skill']} (iş ilanı skoru: {row['skill_score']:.2f}, katkı: +{row['score_increase']:.3f})")


# skill-resources.json dosyasını yükle
with open("../dataset/skill-resources.json", "r", encoding="utf-8") as f:skill_resources = json.load(f)

print("\nKişisel Gelişim Yol Haritası:\n")

career_roadmap = []

for _, row in weighted_df.iterrows():
    skill_name = row["skill"].lower()
    score_inc = row["score_increase"]
    
    if skill_name in skill_resources:
        resources = skill_resources[skill_name]["resources"]
    else:
        query = skill_name.replace(" ", "+")
        resources = [
            {
                "name": f"Udemy – {skill_name} kursları",
                "url": f"https://www.udemy.com/courses/search/?q={query}"
            },
            {
                "name": f"Coursera – {skill_name} eğitimleri",
                "url": f"https://www.coursera.org/search?query={query}"
            }
        ]
    
    career_roadmap.append({
        "skill": skill_name,
        "potential_increase": round(score_inc, 3),
        "resources": resources
    })

for entry in career_roadmap:
    print(f"{entry['skill']} (+{entry['potential_increase']:.3f})")
    for res in entry["resources"]:
        print(f"{res['name']}: {res['url']}")
    print()
    

import joblib
joblib.dump(pipe, "models/logistic_regression_model.pkl")
print("Model kaydedildi: models/logistic_regression_model.pkl")
