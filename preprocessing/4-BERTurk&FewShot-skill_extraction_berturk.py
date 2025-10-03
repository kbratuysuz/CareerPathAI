#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

TURKISH_STOPWORDS = set([
    "ve", "veya", "ile", "için", "bir", "olan", "olmak", "gibi", "vb", "da", "de",
    "olarak", "uygun", "uzerine", "uzerinde", "uzman", "uzmani", "kidemli", 
    "deneyimli", "deneyim", "tecrubeli", "takim", "ekip", "calisma", "sorumluluk",
    "sahip", "iyi", "cok", "en", "gerekli", "aranan", "nitelik", "nitelikler",
    "is", "tanimi", "ilgili", "mezun", "tercihen", "bilgi", "hakim", "hakkinda",
    "seviye", "düzey", "düzeyi", "sahibi", "aranmaktadir"
])

DOMAIN_SEED_SKILLS = [
    ".net", ".net core", "abap", "adobe", "agile", "angular", "api", "asp.net", 
    "aws", "azure", "c#", "csharp", "ci/cd", "ci cd", "cloud", "cloudera", "crm", 
    "css", "cucumber", "design", "devops", "django", "docker", "elasticsearch", 
    "erp", "etl", "express", "figma", "firebase", "flutter", "framework", "gatling", 
    "gcp", "git", "gitlab", "graphql", "hana", "hbase", "hibernate", "html", 
    "hubspot", "illustrator", "java", "javascript", "jenkins", "jira", "jmeter", 
    "kafka", "kubernetes", "laravel", "linux", "matlab", "microservice", "mikroservis", 
    "mongodb", "mssql", "mvc", "mysql", "net", "node", "node.js", "nodejs", "nosql", 
    "oop", "oracle", "patterns", "photoshop", "php", "postgresql", "powerbi", 
    "pytorch", "qliksense", "qlikview", "react", "redis", "rest", "rest api", 
    "salesforce", "sap", "scrum", "selenium", "soap", "spring", "spring boot", 
    "sql", "sqlite", "symfony", "tableau", "tdd", "tensorflow", "test", "testing", 
    "trello", "typescript", "unit test", "unit testing", "vue", "windows", 
    "dotnet", "dotnet core", "angularjs", "sql server", "mlops"
]

# Few-shot examples: (job_description, expected_skills)
FEW_SHOT_EXAMPLES = [
    (
        "java spring boot microservice geliştirme deneyimi aranıyor. mssql veritabanı ve rest api bilgisi gerekli.",
        ["java", "spring boot", "microservice", "mssql", "rest api"]
    ),
    (
        "python django web uygulaması geliştirme. postgresql veritabanı ve docker containerization deneyimi.",
        ["python", "django", "postgresql", "docker"]
    ),
    (
        "react angular frontend geliştirme. typescript javascript bilgisi ve git version control deneyimi.",
        ["react", "angular", "typescript", "javascript", "git"]
    ),
    (
        "aws azure cloud platformları. kubernetes docker container orchestration ve devops süreçleri.",
        ["aws", "azure", "kubernetes", "docker", "devops"]
    ),
    (
        "veri analizi python sql. machine learning pytorch tensorflow ve data science deneyimi.",
        ["python", "sql", "machine learning", "pytorch", "tensorflow", "data science"]
    )
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
            pooled = _mean_pool(outputs.last_hidden_state, enc["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.detach().cpu().numpy())
        return np.vstack(embeddings) if embeddings else np.zeros((0, self.model.config.hidden_size), dtype=np.float32)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return np.matmul(a_norm, b_norm.T)


def _candidate_aliases(skill: str) -> List[str]:
    name = skill.strip().lower()
    aliases: List[str] = [name]
    
    # Common variants
    if name == ".net":
        aliases += ["dotnet", "net"]
    elif name == ".net core" or name == "dotnet core":
        aliases += ["dotnet core", "net core"]
    elif name == "c#":
        aliases += ["csharp", "c sharp"]
    elif name == "node.js" or name == "nodejs":
        aliases += ["nodejs", "node js", "node"]
    elif name == "sql server":
        aliases += ["mssql", "ms sql"]
    elif name == "powerbi":
        aliases += ["power bi"]
    elif name == "machine learning":
        aliases += ["makine öğrenimi", "makine ogrenimi", "ml"]
    elif name == "deep learning":
        aliases += ["derin öğrenme", "derin ogrenme"]
    
    return list(dict.fromkeys(aliases))


def keyword_match_candidates(text: str, candidate_skills: List[str]) -> List[str]:
    """Return candidate names that lexically appear (by alias) in normalized text."""
    text_lc = normalize_text(text)
    matches: List[str] = []
    seen = set()
    
    for c in candidate_skills:
        c_norm = c.strip()
        if not c_norm:
            continue
        key = c_norm.lower()
        if key in seen:
            continue
        
        for alias in _candidate_aliases(c_norm):
            if alias and alias in text_lc:
                matches.append(c_norm)
                seen.add(key)
                break
    
    return matches


def zero_shot_score_skills(
    embedder: BerturkEmbedder,
    text: str,
    candidate_skills: List[str],
    top_n: int = 20,
    min_score: float = 0.0,
) -> List[Dict[str, Any]]:
    if not text:
        return []
    
    normalized = normalize_text(text)
    
    # Deduplicate and filter stopwords
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
    
    # Get embeddings and compute similarities
    text_emb = embedder.embed_texts([normalized])
    cand_embs = embedder.embed_texts(filtered_candidates)
    sims = cosine_similarity_matrix(text_emb, cand_embs)[0]
    
    # Map cosine [-1, 1] to [0, 1]
    scores = (sims + 1.0) / 2.0
    ranked = sorted(zip(filtered_candidates, scores.tolist()), key=lambda x: x[1], reverse=True)
    
    # Intersect with keyword matches in the text
    matched = set(k.lower() for k in keyword_match_candidates(text, [n for n, _ in ranked]))
    intersected = [(n, s) for n, s in ranked if n.lower() in matched]
    
    # Filter by minimum score
    filtered = [(n, s) for n, s in intersected if s >= min_score]
    top = filtered[: top_n if top_n is not None else len(filtered)]
    
    return [{"skill": name, "score": float(round(score, 4))} for name, score in top]


def few_shot_score_skills(
    embedder: BerturkEmbedder,
    text: str,
    candidate_skills: List[str],
    examples: List[Tuple[str, List[str]]] = FEW_SHOT_EXAMPLES,
    top_n: int = 20,
    min_score: float = 0.0,
    few_shot_weight: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Combine zero-shot with few-shot learning using example-based scoring.
    """
    if not text:
        return []
    
    # Get zero-shot scores
    zero_shot_results = zero_shot_score_skills(embedder, text, candidate_skills, top_n=len(candidate_skills), min_score=0.0)
    zero_shot_scores = {item["skill"]: item["score"] for item in zero_shot_results}
    
    # Compute few-shot scores based on example similarity
    normalized_text = normalize_text(text)
    text_emb = embedder.embed_texts([normalized_text])
    
    few_shot_scores = {}
    for example_text, example_skills in examples:
        example_emb = embedder.embed_texts([normalize_text(example_text)])
        similarity = cosine_similarity_matrix(text_emb, example_emb)[0][0]
        
        # Boost scores for skills that appear in similar examples
        for skill in example_skills:
            if skill.lower() in [s.lower() for s in candidate_skills]:
                if skill not in few_shot_scores:
                    few_shot_scores[skill] = 0.0
                few_shot_scores[skill] += similarity * few_shot_weight
    
    # Combine zero-shot and few-shot scores
    combined_scores = {}
    for skill in candidate_skills:
        zero_score = zero_shot_scores.get(skill, 0.0)
        few_score = few_shot_scores.get(skill, 0.0)
        combined_scores[skill] = min(1.0, zero_score + few_score)
    
    # Sort and filter results
    ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    filtered = [(skill, score) for skill, score in ranked if score >= min_score]
    top = filtered[:top_n]
    
    return [{"skill": skill, "score": float(round(score, 4))} for skill, score in top]


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
    min_score: float = 0.0,
    use_few_shot: bool = True,
    few_shot_weight: float = 0.3,
) -> None:
    postings = load_job_postings(input_json, limit=limit)
    embedder = BerturkEmbedder(model_name)
    cand = candidates if candidates is not None else default_skill_candidates()
    
    results: List[Dict[str, Any]] = []
    for item in postings:
        description = item.get("job_description_clean") or item.get("description") or ""
        
        if use_few_shot:
            skills = few_shot_score_skills(
                embedder, description, cand, 
                top_n=top_n, min_score=min_score, few_shot_weight=few_shot_weight
            )
        else:
            skills = zero_shot_score_skills(embedder, description, cand, top_n=top_n, min_score=min_score)
        
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
    
    # Use few-shot learning by default
    process_postings(
        input_path, 
        output_path, 
        model_name=model_name, 
        top_n=20,
        min_score=0.0,
        use_few_shot=True,
        few_shot_weight=0.3
    )


if __name__ == "__main__":
    main()