#!/usr/bin/env python3
import joblib
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, hamming_loss, classification_report
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EvalPrediction
)
from torch.utils.data import Dataset

# ---------------- Step 1: Load preprocessed data ----------------
print("Loading preprocessed data...")
data = joblib.load("./dataset/preprocessed_data.joblib")
train_texts = data["train_texts"]
val_texts = data["val_texts"]
train_labels = data["train_labels"]
val_labels = data["val_labels"]

mlb = joblib.load("./dataset/mlb.joblib")
label_names = mlb.classes_ # numpy array，包含所有分類類別的名稱，順序與 one-hot 編碼中的欄位對應

# ---------------- Step 2: Tokenize ----------------
print("Tokenizing...")

# Use BERT
checkpoint = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(checkpoint)
#  multi-label loss（BCEWithLogitsLoss）
model = BertForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=len(train_labels[0]),  # 對應 multi-label 向量長度
    problem_type="multi_label_classification"
)

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': label
        }

# ---------------- Instantiate datasets ----------------
train_dataset = CustomDataset(train_texts, train_labels, tokenizer)
val_dataset   = CustomDataset(val_texts, val_labels, tokenizer)



# ---------------- Step 4: Move model to GPU if available ----------------
def multi_labels_metrics(predictions, labels, threshold=0.3):
  sigmoid = torch.nn.Sigmoid()
  probs = sigmoid(torch.Tensor(predictions))

  y_pred = np.zeros(probs.shape)
  y_pred[np.where(probs>=threshold)] = 1
  y_true = labels

  f1 = f1_score(y_true, y_pred, average = 'macro')
  roc_auc = roc_auc_score(y_true, y_pred, average = 'macro')
  hamming = hamming_loss(y_true, y_pred)

  metrics = {
      "roc_auc": roc_auc,
      "hamming_loss": hamming,
      "f1": f1
  }

  return metrics

def compute_metrics(p:EvalPrediction):
  preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions

  result = multi_labels_metrics(predictions=preds,
                                labels=p.label_ids)

  return result

# ---------------- Step 5: Training configuration ----------------
training_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    save_steps=1000,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# ---------------- Step 6: Train! ----------------
print("Starting training...")
if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)
    print("Using GPU for training.")

trainer.train()

# ---------------- Step 7: Evaluate ----------------
print("Evaluating...")
preds = trainer.predict(val_dataset).predictions
pred_binary = (preds > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(val_labels, pred_binary, target_names=label_names))
