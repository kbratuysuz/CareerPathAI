# ============================================
# 🧠 Skill Extraction Fine-Tuning (Weighted)
# ============================================

import json
import pandas as pd
import numpy as np
import torch, random
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.model_selection import train_test_split

# --------------------------------------------
# 1️⃣ JSON Verilerini Yükleme
# --------------------------------------------
with open("../../dataset/job-postings/job-posting-dataset.json", "r", encoding="utf-8") as f:
    jobs = {j["job_id"]: j["job_description_clean"] for j in json.load(f)}

with open("../../dataset/job-postings/job-skills.json", "r", encoding="utf-8") as f:
    job_skills = json.load(f)

print("✅ JSON dosyaları yüklendi.")

# Her job_id için {skill: score} dict’i oluştur
label_scores = {
    j["job_id"]: {s["skill"]: s["score"] for s in j["skills"]}
    for j in job_skills
}

# DataFrame oluştur
data = [{"job_id": jid, "text": jobs[jid]} for jid in jobs]
df = pd.DataFrame(data)

# --------------------------------------------
# 2️⃣ Label (Y) Matrisi Oluşturma – Skorlarla
# --------------------------------------------
all_skills = sorted({s for v in [list(v.keys()) for v in label_scores.values()] for s in v})
print(f"🔢 {len(all_skills)} unique skills")

skill2idx = {s: i for i, s in enumerate(all_skills)}
y = np.zeros((len(df), len(all_skills)), dtype=float)

for row_idx, job_id in enumerate(df["job_id"]):
    for skill, score in label_scores.get(job_id, {}).items():
        y[row_idx, skill2idx[skill]] = score  # 1.0–1.06 gibi değerler

print("✅ Skill-score matrisi oluşturuldu:", y.shape)

# --------------------------------------------
# 3️⃣ Model ve Tokenizer
# --------------------------------------------
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(all_skills),
    problem_type="multi_label_classification"
)

# --------------------------------------------
# 4️⃣ Dataset Sınıfı
# --------------------------------------------
class JobPostingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

# --------------------------------------------
# 5️⃣ Eğitim / Doğrulama Split
# --------------------------------------------
texts_train, texts_val, y_train, y_val = train_test_split(
    df["text"].tolist(), y, test_size=0.1, random_state=42
)

train_dataset = JobPostingDataset(texts_train, y_train, tokenizer)
val_dataset = JobPostingDataset(texts_val, y_val, tokenizer)

# --------------------------------------------
# 6️⃣ Eğitim Ayarları
# --------------------------------------------
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

train_args = TrainingArguments(
    output_dir="./outputs",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    #per_device_train_batch_size=8,
    per_device_train_batch_size=4,
    num_train_epochs=6,
    logging_dir="./logs",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# --------------------------------------------
# 7️⃣ Eğitimi Başlat
# --------------------------------------------
trainer.train()

# --------------------------------------------
# 8️⃣ Tahmin ve Sigmoid Skorları
# --------------------------------------------
preds = trainer.predict(train_dataset).predictions
sigmoid = 1 / (1 + np.exp(-preds))
pred_labels = (sigmoid >= 0.1).astype(int)

# --------------------------------------------
# 9️⃣ Sonuçları JSON’a Kaydet
# --------------------------------------------
output = []
for idx in range(10):
    job_id = df.loc[idx, "job_id"]
    probs = sigmoid[idx]
    predicted_skills = [all_skills[i] for i in np.where(pred_labels[idx] == 1)[0]]
    skill_scores = {
        all_skills[i]: float(probs[i])
        for i in np.where(pred_labels[idx] == 1)[0]
    }

    output.append({
        "job_id": job_id,
        "predicted_skills": predicted_skills,
        "sigmoid_scores": skill_scores
    })

with open("prediction-results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("✅ Tahmin sonuçları 'prediction-results.json' dosyasına kaydedildi.")

# --------------------------------------------
# 🔍 Ek: Sigmoid Dağılım Görselleştirme
# --------------------------------------------
import matplotlib.pyplot as plt
plt.hist(sigmoid.flatten(), bins=50)
plt.title("Sigmoid Probability Distribution (Weighted Labels)")
plt.xlabel("Probability")
plt.ylabel("Frequency")
plt.show()
