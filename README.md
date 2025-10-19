# 💼 CareerPathAI

**CareerPathAI** is an AI-powered **Career Compatibility and Skill Development System** designed to semantically match **CVs** with **job postings**, identify **missing skills**, and generate **personalized learning roadmaps**.  
It leverages **Turkish NLP models**, **machine learning**, and **Streamlit UI** to support students and professionals in career planning, skill improvement, and job-market alignment.

---

## 🚀 Key Features

- **AI-based Job–CV Matching**  
  Matches candidates and job postings using Logistic Regression and Random Forest models trained on structured skill data.

- **Skill Extraction Pipeline (Turkish NLP)**  
  Extracts technical and soft skills from job descriptions using multiple methods:
  - **Regex + Gazetteer Hybrid**
  - **KeyBERT-based Extraction**
  - **BERTurk Zero-Shot & Few-Shot Models**

- **Career Roadmap Generator**  
  Detects missing skills for each candidate and recommends **Udemy**, **Coursera**, and other resources for improvement.

- **Interactive Web Interface (Streamlit)**  
  Provides user authentication, CV creation wizard, skill-based career matching dashboard, and detailed job analysis pages.

---

## 🧠 System Architecture

Dataset (CVs + Job Postings)

│

▼
Skill Extraction Layer (Regex, KeyBERT, BERTurk)

│

▼
Skill Matching Models

├── Logistic Regression

├── Random Forest

│

▼
Match Scoring & Roadmap Generation

│

▼
Streamlit Web Application

---

## 🧩 Model Components

| Module | Description |
|--------|--------------|
| `1-KeyBERT_based_skill_extraction_berturk.py` | Extracts candidate skills using multilingual sentence embeddings. |
| `2-BERTurk&ZeroShot-skill_extraction_berturk.py` | Uses **BERTurk** embeddings for zero-shot skill scoring. |
| `3-regex-gazetteer-combination-skill-extraction.py` | Hybrid rule-based extractor combining regex patterns and gazetteer dictionaries. |
| `4-BERTurk&FewShot-skill_extraction_berturk.py` | Enhances zero-shot results using few-shot contextual examples. |
| `logistic-regression-matching.py` | Computes job–CV compatibility scores and missing skill impacts. |
| `random-forest-matching.py` | Alternative ensemble-based matching model for robustness. |
| `model.py` | Fine-tunes **XLM-Roberta** for multi-label skill classification. |

---

## 🖥️ Streamlit Interface Modules

| Page | Purpose |
|------|----------|
| `app.py` | Main entry point and navigation controller. |
| `login_page.py` / `register_page.py` | User authentication system. |
| `cv_input_wizard.py` | Step-by-step CV builder with education, skills, and experience inputs. |
| `profile_page.py` | Displays and edits saved CV data. |
| `job_matches_page.py` | Lists top job matches for the logged-in user. |
| `job_detail_page.py` | Shows skill alignment, missing skills, and a personalized development roadmap. |

---

## 🧰 Tech Stack

- **Language:** Python 3.11
- **Libraries:**  
  `transformers`, `sentence-transformers`, `torch`, `scikit-learn`, `keybert`, `streamlit`, `pandas`, `matplotlib`
- **Modeling:** Logistic Regression, Random Forest, BERTurk, XLM-Roberta  
- **Interface:** Streamlit  
- **Storage:** JSON-based lightweight datasets for CVs, users, and jobs

---

## 📦 Installation

```bash
git clone https://github.com/kbratuysuz/CareerPathAI.git
cd CareerPathAI
pip install -r requirements.txt
```

---

## ▶️ Running the Application

```bash
streamlit run app.py
```
Then open your browser and navigate to http://localhost:8501

---

## 📚 Dataset Structure

dataset/

├── resumes/

│   └── cv-dataset-all.json

├── job-postings/

│   ├── job-posting-dataset-all.json

│   ├── job-skills-all.json

│   └── job-posting-skills.json

├── skill-resources.json

└── skill-list-all.json

---

## 🧭 Example Workflow


1. User registers and logs in.

2. Uploads or builds a CV through the wizard interface.

3. The system analyzes job postings and calculates match scores.

4. Missing skills are identified and weighted by job relevance.

5. A personalized learning roadmap is generated with resource links.

---

## 🧩 Future Improvements

Integration of Graph Neural Networks (GNN) for career path prediction

Explainable AI (XAI) modules for skill relevance insights

Reinforcement Learning for adaptive career recommendations

Multi-language support and API integration for HR systems

---

## 👩‍💻 Authors

Developed by Büşra Eşkara, Kübra Tüysüz Aksu, and Şevval Yıldız as part of the AI-Powered Career Compatibility and Development System Project (GYK).

---

## 🌟 Acknowledgments

This project leverages open-source NLP models from Hugging Face and builds upon academic work in skill extraction, semantic matching, and career analytics.

---
