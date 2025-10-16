from pathlib import Path
import json
from turtle import st

@st.cache_data(show_spinner=False)
def get_university_department_map():
    path = Path("dataset/resumes/cv-dataset-all.json")
    uni_dept_map = {}

    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for record in data:
                edu = record.get("education", {})
                for uni, dept in edu.items():
                    uni_clean = uni.strip().title() 
                    dept_clean = dept.strip().title()

                    if uni_clean not in uni_dept_map:
                        uni_dept_map[uni_clean] = set()
                    uni_dept_map[uni_clean].add(dept_clean)

            uni_dept_map = {u: sorted(list(deps)) for u, deps in uni_dept_map.items()}

        except Exception as e:
            print("Veri okunamadı:", e)

    return uni_dept_map

@st.cache_data(show_spinner=False)
def get_roles_from_datasets():
    cv_path = Path("dataset/resume/cv-dataset-all.json")
    job_path = Path("dataset/job-postings/job-posting-dataset-all.json")

    roles = set()

    if cv_path.exists():
        try:
            with open(cv_path, "r", encoding="utf-8") as f:
                cv_data = json.load(f)
            for record in cv_data:
                experiences = record.get("experience", [])
                if isinstance(experiences, list):
                    for exp in experiences:
                        role = exp.get("role")
                        if role and isinstance(role, str):
                            roles.add(role.strip().title())
        except Exception as e:
            print("CV dataset okunamadı:", e)

    if job_path.exists():
        try:
            with open(job_path, "r", encoding="utf-8") as f:
                job_data = json.load(f)
            for job in job_data:
                title = job.get("job_title_clean")
                if title and isinstance(title, str):
                    roles.add(title.strip().title())
        except Exception as e:
            print("Job dataset okunamadı:", e)

    return sorted(list(roles))
