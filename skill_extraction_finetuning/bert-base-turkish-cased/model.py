import json
import pandas as pd

with open("job_posting_dataset.json", "r", encoding="utf-8") as f:
    jobs = {j["job_id"]: j["job_description_clean"] for j in json.load(f)}

with open("weak_labels.json", "r", encoding="utf-8") as f:
    labels = {j["job_id"]: [s["skill"] for s in j["skills"]] for j in json.load(f)}

data = [{"job_id": jid, "text": jobs[jid], "labels": labels.get(jid, [])} for jid in jobs]
df = pd.DataFrame(data)

all_skills = sorted({s for v in labels.values() for s in v})
print(len(all_skills), "unique skills")

from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer(classes=all_skills)
y = mlb.fit_transform(df["labels"])

from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_name = "dbmdz/bert-base-turkish-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(all_skills),
    problem_type="multi_label_classification"
)

import torch
from torch.utils.data import Dataset

class JobPostingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
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

from transformers import Trainer, TrainingArguments
import torch
import numpy as np

train_args = TrainingArguments(
    output_dir="./outputs",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    num_train_epochs=4,
    logging_dir="./logs",
    load_best_model_at_end=True,
)

train_dataset = JobPostingDataset(df["text"].tolist(), y, tokenizer)

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset
)

trainer.train()

preds = trainer.predict(train_dataset).predictions
sigmoid = 1 / (1 + np.exp(-preds))
pred_labels = (sigmoid >= 0.1).astype(int)

for idx in range(10):
    job_id = df.loc[idx, "job_id"]
    skills = mlb.classes_
    probs = sigmoid[idx]

    predicted_skills = mlb.inverse_transform(pred_labels[[idx]])[0]

    skill_scores = {skills[i]: round(probs[i], 3) for i in np.where(pred_labels[idx] == 1)[0]}

    print(f"\nJob ID: {job_id}")
    print(f"Predicted Skills: {predicted_skills}")
    print(f"Sigmoid Scores: {skill_scores}")

import matplotlib.pyplot as plt

plt.hist(sigmoid.flatten(), bins=50)
plt.title("Sigmoid Probability Distribution")
plt.xlabel("Probability")
plt.ylabel("Frequency")
plt.show()