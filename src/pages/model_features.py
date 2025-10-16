import pandas as pd

def build_features_for_match(cv_skills, job_skills):
    cv_skills = set([s.lower() for s in cv_skills])
    job_skills = set([j["skill"].lower() for j in job_skills])

    intersection = cv_skills & job_skills
    missing = job_skills - cv_skills
    union = cv_skills | job_skills

    features = {
        "skill_match_count": len(intersection),
        "skill_match_ratio": len(intersection) / len(job_skills) if job_skills else 0,
        "cv_skill_count": len(cv_skills),
        "job_skill_count": len(job_skills),
        "missing_skill_count": len(missing),
        "jaccard_similarity": len(intersection) / len(union) if union else 0
    }

    return pd.DataFrame([features])
