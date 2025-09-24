import pandas as pd
import re
import unicodedata

EXCEL_PATH = "dataset/job-postings/jobs_kariyernet.xlsx"
df = pd.read_excel(EXCEL_PATH)

df = df.dropna(how="all")
df = df.drop_duplicates()

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()

    text = unicodedata.normalize("NFKC", text)
    text = "".join(
        ch if unicodedata.category(ch).startswith(("L", "N")) or ch.isspace() else " "
        for ch in text
    )

    # text = text.replace("•", " ").replace("·", " ").replace("“", " ").replace("”", " ").replace("-", " ")  
    # text = text.translate(str.maketrans("", "", string.punctuation))

    text = re.sub(r"\s+", " ", text).strip()
    return text

def parse_app_count(val):
    if pd.isna(val):
        return None
    val = str(val)
    match = re.search(r"(\d+)", val)
    return int(match.group(1)) if match else None

def clear_dataset() : 
    text_cols = [
        "job_title",
        "company_name",
        "location",
        "employment_type",
        "position_level",
        "department",
        "application_count",
        "additional_info",
        "job_description",
    ]

    for col in text_cols:
        if col in df.columns:
            df[col + "_clean"] = df[col].apply(clean_text)

    if "application_count" in df.columns:
        df["application_count_num"] = df["application_count"].apply(parse_app_count)


    OUTPUT_PATH = "dataset/job-postings/jobs_kariyernet_clean.xlsx"
    df.to_excel(OUTPUT_PATH, index=False)

    print("dataset has been cleaned successfully")

import pandas as pd
from deep_translator import GoogleTranslator

def translate_all_cells() :
    df = pd.read_excel("dataset\job-postings\jobs_kariyernet_clean.xlsx")

    df_tr = df.applymap(lambda x: GoogleTranslator(source='en', target='tr').translate(str(x)) if isinstance(x, str) else x)

    df_tr.to_excel("dataset\job-postings\jobs_kariyernet_clean_translated.xlsx", index=False)

# clear_dataset()

translate_all_cells()