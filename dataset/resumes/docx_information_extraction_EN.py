import os
import re
import json
from docx import Document

def extract_text(docx_path):
    doc = Document(docx_path)
    return "\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip() != ""])

def extract_certifications(text):
    known = ["PMP", "Six Sigma", "SMC", "Scrum Master", "CSM", "PMI"]
    return [c for c in known if c.lower() in text.lower()]

def extract_skills(text):
    keywords = ["Agile", "Scrum", "Spring", "Java", "SAP", "AWS", "JIRA", "Oracle", "MS Project", "SharePoint", "Visio", "Service Now"]
    return list({k for k in keywords if k.lower() in text.lower()})

def extract_education(text):
    return [line for line in text.split("\n") if "university" in line.lower() or "school" in line.lower() or "degree" in line.lower() or "bachelor" in line.lower()]

def extract_experience_blocks(text):
    blocks = []
    pattern = re.compile(r"(.*?)\s+(?:\d{1,2}/\d{4}\s*[-–]\s*\d{1,2}/\d{4}|Present)", re.DOTALL)
    matches = pattern.finditer(text)
    for m in matches:
        company_line = m.group(1).strip()
        block_start = m.end()
        block_end = text.find("\n\n", block_start)
        block = text[block_start:block_end].strip() if block_end > 0 else text[block_start:].strip()
        blocks.append({
            "company_or_title": company_line,
            "description": block
        })
    return blocks

def process_cv(docx_path, cv_id):
    text = extract_text(docx_path)
    return {
        "cv_id": cv_id,
        "skills": extract_skills(text),
        "certifications": extract_certifications(text),
        "education": extract_education(text),
        "experience": extract_experience_blocks(text)
    }

docx_files = [
    ("Adhi Gopalam - SM.docx", "cv_001"),
    ("Achyuth Resume_8.docx", "cv_002"),
    ("Adelina_Erimia_PMP1.docx", "cv_003")
]

output = []
for file_name, cv_id in docx_files:
    file_path = f"/mnt/data/{file_name}"
    output.append(process_cv(file_path, cv_id))

with open("cv_dataset.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
