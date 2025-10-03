import json
import os
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
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
    ".net",".net core", "abap", "adobe", "agile", "angular", "api", "asp.net", "aws", "azure",
    "c#", "csharp", "ci/cd", "ci cd", "cloud", "cloudera", "crm", "css", "cucumber", "design", "devops",
    "django", "docker", "elasticsearch", "erp", "etl", "express", "figma", "firebase",
    "flutter", "framework", "gatling", "gcp", "git", "gitlab", "graphql", "hana",
    "hbase", "hibernate", "html", "hubspot", "illustrator", "java", "javascript",
    "jenkins", "jira", "jmeter", "kafka", "kubernetes", "laravel", "linux", "matlab",
    "microservice", "mikroservis", "mongodb", "mssql", "mvc", "mysql", "net", "node",
    "node.js", "nodejs", "nosql", "oop", "oracle", "patterns", "photoshop", "php",
    "postgresql", "powerbi", "pytorch", "qliksense", "qlikview", "react", "redis",
    "rest", "rest api", "salesforce", "sap", "scrum", "selenium", "soap", "spring",
    "spring boot", "sql", "sqlite", "symfony", "tableau", "tdd", "tensorflow", "test",
    "testing", "trello", "typescript", "unit test", "unit testing", "vue", "windows", "dotnet", "dotnet core",
    "angularjs", "mysql", "postgresql", "sql server", "mlops"
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


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask_expanded, dim=1)
    counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return summed / counts


class BerturkEmbedder:
    def __init__(self, model_name: str = "dbmdz/bert-base-turkish-cased", device: Optional[str] = None) -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed_texts(self, texts: List[str], batch_size: int = 16, max_length: int = 256) -> np.ndarray:
        embeddings: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            outputs = self.model(**enc)
            pooled = _mean_pool(outputs.last_hidden_state, enc["attention_mask"])  # [B, H]
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.detach().cpu().numpy())
        return np.vstack(embeddings) if embeddings else np.zeros((0, self.model.config.hidden_size), dtype=np.float32)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return np.matmul(a_norm, b_norm.T)


def zero_shot_score_skills(
    embedder: BerturkEmbedder,
    text: str,
    candidate_skills: List[str],
    top_n: int = 20,
) -> List[Dict[str, Any]]:
    if not text:
        return []
    normalized = normalize_text(text)

    # Deduplicate and keep order
    seen = set()
    filtered_candidates: List[str] = []
    for c in candidate_skills:
        c_norm = c.strip()
        if not c_norm:
            continue
        if c_norm.lower() in TURKISH_STOPWORDS or c_norm.lower() in ENGLISH_STOP_WORDS:
            continue
        if c_norm.lower() in seen:
            continue
        seen.add(c_norm.lower())
        filtered_candidates.append(c_norm)

    if not filtered_candidates:
        return []

    text_emb = embedder.embed_texts([normalized])  # [1, H]
    cand_embs = embedder.embed_texts(filtered_candidates)  # [C, H]
    sims = cosine_similarity_matrix(text_emb, cand_embs)[0]  # [C]

    # Map cosine [-1, 1] to [0, 1]
    scores = (sims + 1.0) / 2.0
    ranked = sorted(zip(filtered_candidates, scores.tolist()), key=lambda x: x[1], reverse=True)
    top = ranked[: top_n if top_n is not None else len(ranked)]
    return [{"skill": name, "score": float(round(score, 4))} for name, score in top]


def default_skill_candidates() -> List[str]:
    common = [
        "Python", "Java", "C#", "C++", "JavaScript", "TypeScript", "SQL", "NoSQL", "React",
        "Angular", "Vue", "Django", "Flask", "Spring", "Spring Boot", "Node.js", "ASP.NET",
        ".NET", "Docker", "Kubernetes", "AWS", "Azure", "GCP", "Linux", "Git", "CI/CD",
        "Microservice", "Mikroservis", "REST", "GraphQL", "Elasticsearch", "Redis", "PostgreSQL",
        "MySQL", "MongoDB", "Oracle", "PowerBI", "Tableau", "TensorFlow", "PyTorch",
        "MLOps", "Data Analysis", "ETL", "Unit Testing", "Communication", "Teamwork", "Problem Solving",
    ]
    dedup: List[str] = []
    seen = set()
    for s in DOMAIN_SEED_SKILLS + common:
        key = s.strip().lower()
        if key and key not in seen:
            seen.add(key)
            dedup.append(s)
    return dedup


def process_postings(
    input_json: str,
    output_json: str,
    model_name: str = "dbmdz/bert-base-turkish-cased",
    limit: Optional[int] = None,
    candidates: Optional[List[str]] = None,
    top_n: int = 20,
) -> None:
    postings = load_job_postings(input_json, limit=limit)
    embedder = BerturkEmbedder(model_name)
    cand = candidates if candidates is not None else default_skill_candidates()

    results: List[Dict[str, Any]] = []
    for item in postings:
        description = item.get("job_description_clean") or item.get("description") or ""
        skills = zero_shot_score_skills(embedder, description, cand, top_n=top_n)
        results.append({
            "job_id": item.get("job_id") or item.get("id") or item.get("_id"),
            "skills": skills,
        })

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    input_path = os.path.join(base_dir, "dataset", "job-postings", "job-posting-dataset.json")
    output_path = os.path.join(base_dir, "dataset", "job-postings", "job-posting-skills.json")

    model_name = "dbmdz/bert-base-turkish-cased"
    process_postings(input_path, output_path, model_name=model_name, top_n=20)


if __name__ == "__main__":
    main()


