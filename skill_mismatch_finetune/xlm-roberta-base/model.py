# job-posting-dataset.json + gazetteer çıktısını birleştirir
import json
import pandas as pd

with open("dataset/job_posting_dataset.json", "r", encoding="utf-8") as f:
    jobs = {j["job_id"]: j["job_description_clean"] for j in json.load(f)}

with open("dataset/weak_labels.json", "r", encoding="utf-8") as f:
    labels = {j["job_id"]: [s["skill"] for s in j["skills"]] for j in json.load(f)}

data = [{"job_id": jid, "text": jobs[jid], "labels": labels.get(jid, [])} for jid in jobs]
df = pd.DataFrame(data)

# unique skill list
all_skills = sorted({s for v in labels.values() for s in v})
print(len(all_skills), "unique skills")

# encode labels step
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer(classes=all_skills)
y = mlb.fit_transform(df["labels"])

# model selection step
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# model_name = "dbmdz/bert-base-turkish-cased"
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(all_skills),
    problem_type="multi_label_classification"
)

# preparing dataset step
import torch
from torch.utils.data import Dataset

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

# fine-tuning step
from transformers import Trainer, TrainingArguments
import numpy as np, random
from sklearn.model_selection import train_test_split

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

train_dataset = JobPostingDataset(df["text"].tolist(), y, tokenizer)

texts_train, texts_val, y_train, y_val = train_test_split(
    df["text"].tolist(), y, test_size=0.1, random_state=42
)

train_dataset = JobPostingDataset(texts_train, y_train, tokenizer)
val_dataset = JobPostingDataset(texts_val, y_val, tokenizer)

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

# evaluate step
preds = trainer.predict(train_dataset).predictions
sigmoid = 1 / (1 + np.exp(-preds))
pred_labels = (sigmoid >= 0.1).astype(int)

# Örnek: İlan bazında skill tahmini
for idx in range(10):
    job_id = df.loc[idx, "job_id"]
    skills = mlb.classes_
    probs = sigmoid[idx]

    # tahmin edilen skiller
    predicted_skills = mlb.inverse_transform(pred_labels[[idx]])[0]

    # sadece tahmin edilen skillerin skorlarını al
    skill_scores = {skills[i]: round(probs[i], 3) for i in np.where(pred_labels[idx] == 1)[0]}

    print(f"\n🧩 Job ID: {job_id}")
    print(f"🎯 Predicted Skills: {predicted_skills}")
    print(f"📊 Sigmoid Scores: {skill_scores}")

import matplotlib.pyplot as plt

plt.hist(sigmoid.flatten(), bins=50)
plt.title("Sigmoid Probability Distribution")
plt.xlabel("Probability")
plt.ylabel("Frequency")
plt.show()