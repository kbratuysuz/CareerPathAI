import json
import os
from typing import List, Dict, Any, Optional

import numpy as np
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from unidecode import unidecode

TURKISH_STOPWORDS = set(
    [
        # Common Turkish stopwords and noisy tokens often seen in postings
        "ve",
        "veya",
        "ile",
        "için",
        "i\u00e7in",
        "bir",
        "olan",
        "olmak",
        "gibi",
        "vb",
        "da",
        "de",
        "ile",
        "ile",
        "olarak",
        "uygun",
        "uzerine",
        "uzerinde",
        "uzman",
        "uzmani",
        "kidemli",
        "deneyimli",
        "deneyim",
        "tecrubeli",
        "tecr\u00fcbeli",
        "takim",
        "ekip",
        "calisma",
        "sorumluluk",
        "sahip",
        "iyi",
        "cok",
        "en",
        "gerekli",
        "aranan",
        "nitelik",
        "nitelikler",
        "is",
        "i\u015f",
        "tanimi",
        "tan\u0131m\u0131",
        "ilgili",
        "mezun",
        "tercihen",
        "bilgi",
        "hakim",
        "h\u00e2kim",
        "hakkinda",
        "hakk\u0131nda",
        "uzerinde",
        "seviye",
        "d\u00fczey",
        "d\u00fczeyi",
        "sahibi",
        "sahip",
        "aranmaktadir",
        "aranmaktad\u0131r",
    ]
)

DOMAIN_SEED_SKILLS = [
    # Software / data common skills for better precision (expand later)
    "python",
    "java",
    "c#",
    ".net",
    "net core",
    "asp.net",
    "react",
    "angular",
    "vue",
    "javascript",
    "typescript",
    "node.js",
    "nodejs",
    "express",
    "spring",
    "spring boot",
    "hibernate",
    "sql",
    "postgresql",
    "mysql",
    "mssql",
    "oracle",
    "mongodb",
    "redis",
    "kafka",
    "elasticsearch",
    "docker",
    "kubernetes",
    "microservice",
    "mikroservis",
    "rest api",
    "graphql",
    "git",
    "ci/cd",
    "jenkins",
    "gitlab ci",
    "unit testing",
    "tdd",
    "agile",
    "scrum",
    "flutter",
    "firebase",
    "aws",
    "azure",
    "gcp",
    "powerbi",
    "tableau",
]

def load_job_postings(json_path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if limit is not None:
        data = data[:limit]
    return data


def normalize_text(text: str) -> str:
    if not text:
        return ""
    # Keep diacritics for Turkish model; create a secondary ascii form for filtering if needed
    # Here we only lowercase and strip extra spaces.
    return " ".join(text.lower().split())


def build_kw_model(model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") -> KeyBERT:
    # A lightweight multilingual model that works well on Turkish. For BERTürk you can switch to e.g. dbmdz/bert-base-turkish-cased
    # but KeyBERT needs sentence embeddings; using a SentenceTransformer wrapper is recommended.
    st_model = SentenceTransformer(model_name)
    return KeyBERT(model=st_model)


def extract_skills(
    kw_model: KeyBERT,
    text: str,
    top_n: int = 15,
    max_ngram: int = 3,
    use_mmr: bool = True,
    diversity: float = 0.4,
) -> List[Dict[str, Any]]:
    if not text:
        return []

    normalized = normalize_text(text)

    # Combine Turkish and English stopwords to filter mixed postings
    stop_words = TURKISH_STOPWORDS.union(ENGLISH_STOP_WORDS)

    try:
        keywords = kw_model.extract_keywords(
            normalized,
            keyphrase_ngram_range=(1, max_ngram),
            stop_words=list(stop_words),
            use_mmr=use_mmr,
            diversity=diversity,
            top_n=top_n,
            seed_keywords=DOMAIN_SEED_SKILLS,
        )
    except Exception:
        keywords = []

    results: List[Dict[str, Any]] = []
    for kw, score in keywords:
        # Basic cleanup
        token = kw.strip()
        if not token or token.isdigit():
            continue
        # Filter obvious non-skill fillers
        if token in stop_words:
            continue
        results.append({"skill": token, "score": float(score)})

    return results


def process_postings(
    input_json: str,
    output_json: str,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    limit: Optional[int] = None,
) -> None:
    postings = load_job_postings(input_json, limit=limit)
    kw_model = build_kw_model(model_name)

    enriched: List[Dict[str, Any]] = []
    for item in postings:
        description = item.get("job_description_clean") or item.get("description") or ""
        skills = extract_skills(kw_model, description)
        enriched.append({**item, "extracted_skills": skills})

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)


def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    input_path = os.path.join(base_dir, "dataset", "job-postings", "job-posting-dataset.json")
    output_path = os.path.join(base_dir, "dataset", "job-postings", "job-posting-skills.json")

    # You can switch to a BERTürk SentenceTransformer, e.g.: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # Alternatives for Turkish: "paraphrase-multilingual-mpnet-base-v2" or a distilled Turkish model if available.
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    process_postings(input_path, output_path, model_name=model_name)


if __name__ == "__main__":
    main()


