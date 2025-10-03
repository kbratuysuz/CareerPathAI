#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import re
import argparse
from pathlib import Path

# ---------- 1) Technical Skill Synonyms ----------
TECHNICAL_SYNONYMS = {
    # Programlama Dilleri
    "java": ["java"],
    "javascript": ["javascript", "js"],
    "typescript": ["typescript", "ts"],
    "python": ["python", "py"],
    "c#": ["c#", "c sharp", "csharp"],
    "c++": ["c++", "cpp"],
    "c": [r"\bc\b", "c language"],

    # .NET
    ".net": [".net", "dotnet", "net framework", "asp.net", "aspnet"],
    ".net core": [".net core", "dotnet core", "net core"],

    # Frameworkler
    "spring": ["spring", "spring boot"],
    "hibernate": ["hibernate"],
    "react": ["react", "reactjs", "react.js"],
    "angular": ["angular", "angularjs", "angular.js"],
    "vue": ["vue", "vuejs", "vue.js", "nuxt", "nuxt.js", "nuxtjs"],
    "node.js": ["node.js", "nodejs", "node js"],
    "express": ["express", "expressjs", "express.js"],
    "django": ["django"],
    "flask": ["flask"],
    "fastapi": ["fastapi"],
    "laravel": ["laravel"],
    "symfony": ["symfony"],
    "flutter": ["flutter"],

    # Veri & Database
    "data": ["veri", "büyük veri", "big data", "veri analizi", "data analysis"],
    "database": ["veritabanı", "database", "db"],
    "sql": [r"(?<!no)sql", "sql sorgulama"],
    "sql server": ["sql server", "mssql", "ms sql", "ms-sql"],
    "mysql": ["mysql", "my sql"],
    "postgresql": ["postgresql", "postgres", "psql"],
    "oracle": ["oracle", "pl/sql", "plsql"],
    "mongodb": ["mongodb", "mongo db"],
    "redis": ["redis"],
    "elasticsearch": ["elasticsearch", "elk"],
    "kafka": ["kafka", "apache kafka"],
    "network": ["network", "ağ yönetimi", "ağ teknolojileri"],
    "erp": ["erp", "erp sistemleri"],

    # Güvenlik
    "security": ["güvenlik", "siber güvenlik", "information security"],

    # Test
    "testing": ["test", "yazılım testi", "test süreçleri"],
    "unit testing": ["unit testing", "unit test", "unittest", "unit-test"],
    "selenium": ["selenium"],
    "jmeter": ["jmeter"],

    # DevOps & Cloud
    "docker": ["docker"],
    "kubernetes": ["kubernetes", "k8s"],
    "git": ["git"],
    "jenkins": ["jenkins"],
    "ci cd": ["ci/cd", "ci cd", "cicd"],
    "devops": ["devops"],
    "aws": ["aws", "amazon web services"],
    "azure": ["azure", "microsoft azure"],
    "gcp": ["gcp", "google cloud"],
    "cloud": ["cloud", "bulut"],

    # Raporlama
    "reporting": ["raporlama", "reporting"]
}

# ---------- 2) Soft & Business Skills Synonyms ----------
SOFT_BUSINESS_SYNONYMS = {
    "communication": ["iletişim", "iletişim becerileri", "takım içi iletişim"],
    "analytical thinking": ["analitik", "analitik düşünme", "analiz yapma"],
    "problem solving": ["problem çözme", "çözüm üretme", "çözme"],
    "teamwork": ["takım çalışması", "ekip çalışması", "işbirliği"],
    "leadership": ["liderlik", "ekip yönetimi", "yönetim becerileri"],
    "customer focus": ["müşteri odaklı", "müşteri ilişkileri", "müşteri yönetimi"],
    "adaptability": ["yenilikçi", "esnek", "değişime açık"],
    "responsibility": ["sorumluluk", "görev bilinci"],
    "attention to detail": ["dikkatli", "detay odaklı"],
    "performance": ["performans", "yüksek performans"],
    "support": ["destek", "destek hizmetleri"],
    "organization": ["organizasyon", "planlama", "yönetim"]
}

# Combine both
ALL_SYNONYMS = {**TECHNICAL_SYNONYMS, **SOFT_BUSINESS_SYNONYMS}

# ---------- 3) Regex Yardımcıları ----------
def _build_pattern(variant: str) -> re.Pattern:
    v = variant.lower()
    if variant == r"(?<!no)sql":
        return re.compile(variant, re.IGNORECASE)
    v = v.replace(".", r"\.")
    v = v.replace("+", r"\+")
    v = v.replace("#", r"\#")
    v = v.replace(" ", r"[ \._\-]?")
    return re.compile(rf"(?<![A-Za-z0-9\+\.#]){v}(?![A-Za-z0-9\+\.#])", re.IGNORECASE)

def compile_patterns(synonyms: dict) -> dict:
    compiled = {}
    for canon, variants in synonyms.items():
        patterns = []
        for v in set([canon] + variants):
            try:
                patterns.append(_build_pattern(v))
            except re.error:
                patterns.append(re.compile(re.escape(v), re.IGNORECASE))
        compiled[canon] = patterns
    return compiled

COMPILED = compile_patterns(ALL_SYNONYMS)

# ---------- 4) Çıkarım Fonksiyonu ----------
def extract_skills(text: str, compiled=COMPILED, topk: int = 30):
    t = (text or "").lower()
    found = {}
    for canon, patterns in compiled.items():
        hit = 0
        for p in patterns:
            if p.search(t):
                hit += 1
        if hit > 0:
            found[canon] = round(1.0 + min(hit, 3) * 0.02, 3)
    items = sorted(found.items(), key=lambda x: (-x[1], x[0]))
    return [{"skill": k, "score": v} for k, v in items[:topk]]

# ---------- 5) I/O ----------
def run(input_path: Path, output_path: Path, topk: int = 30):
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    out = []
    for row in data:
        job_id = row.get("job_id")
        desc = row.get("job_description_clean") or row.get("job_description") or ""
        skills = extract_skills(desc, COMPILED, topk=topk)
        out.append({"job_id": job_id, "skills": skills})
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(out)} iş ilanı işlendi. Çıktı: {output_path}")

# ---------- 6) CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default="dataset/job-postings/job-posting-dataset.json")
    ap.add_argument("--output", type=Path, default="dataset/job-postings/job-posting-skills.json")
    ap.add_argument("--topk", type=int, default=30)
    args = ap.parse_args()
    run(args.input, args.output, args.topk)
