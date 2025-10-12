import json
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments
import numpy as np, random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


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
    
with open("dataset/job_posting_dataset.json", "r", encoding="utf-8") as f:
    jobs = {j["job_id"]: j["job_description_clean"] for j in json.load(f)}

with open("dataset/weak_labels.json", "r", encoding="utf-8") as f:
    labels = {j["job_id"]: [s["skill"] for s in j["skills"]] for j in json.load(f)}

data = [{"job_id": jid, "text": jobs[jid], "labels": labels.get(jid, [])} for jid in jobs]
df = pd.DataFrame(data)

all_skills = sorted({s for v in labels.values() for s in v})
print(len(all_skills), "unique skills")

mlb = MultiLabelBinarizer(classes=all_skills)
y = mlb.fit_transform(df["labels"])

# model_name = "dbmdz/bert-base-turkish-cased"
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(all_skills),
    problem_type="multi_label_classification"
)

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

preds = trainer.predict(train_dataset).predictions
sigmoid = 1 / (1 + np.exp(-preds))
pred_labels = (sigmoid >= 0.1).astype(int)

output = []

for idx in range(10):
    job_id = df.loc[idx, "job_id"]
    skills = mlb.classes_
    probs = sigmoid[idx]

    predicted_skills = mlb.inverse_transform(pred_labels[[idx]])[0]
    skill_scores = {
        skills[i]: float(probs[i]) 
        for i in np.where(pred_labels[idx] == 1)[0]
    }

    output.append({
        "job_id": job_id,
        "sigmoid_scores": skill_scores
    })

with open("prediction-results.json", "w", encoding="utf-8") as f: json.dump(output, f, ensure_ascii=False, indent=2)

print("✅ Tahmin sonuçları 'predictions.json' dosyasına kaydedildi.")


plt.hist(sigmoid.flatten(), bins=50)
plt.title("Sigmoid Probability Distribution")
plt.xlabel("Probability")
plt.ylabel("Frequency")
plt.show()