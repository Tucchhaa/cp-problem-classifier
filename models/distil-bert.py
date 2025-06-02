import pandas as pd, numpy as np, torch, evaluate, random, os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, hamming_loss, precision_recall_curve
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from data_manager.utils import get_dataset_filepath
import ast
from datasets import Dataset
from transformers import pipeline
import pickle

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Prepare Dataset
problems_df = pd.read_csv(get_dataset_filepath("problems.csv"))
problems_df['labels'] = problems_df['labels'].apply(ast.literal_eval)

# Binarize labels
mlb      = MultiLabelBinarizer()
y_matrix = mlb.fit_transform(problems_df['labels'])
problems_df['label_vec'] = list(y_matrix)
NUM_LABELS = y_matrix.shape[1]


# Initialize model
MODEL_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

config = AutoConfig.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    problem_type="multi_label_classification"   # tells HF to use BCEWithLogits
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, config=config
)

# Compute class weights
#  pos_weight = (#negative / #positive)  for each label
label_pos = y_matrix.sum(axis=0)
label_neg = y_matrix.shape[0] - label_pos
# avoid division-by-zero; if a class never appears, give it weight 0
# pos_weight = np.where(label_pos == 0, 0.0, label_neg / label_pos)
# pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
ratio = torch.tensor(label_neg / label_pos, dtype=torch.float32)
pos_weight = torch.clamp(torch.log1p(ratio), min=1.0, max=10.0)

# Split train/test datasets
train_df, val_df = train_test_split(
    problems_df, test_size=0.2, random_state=SEED, shuffle=True
)

def tokenize_batch(batch):
    encodings = tokenizer(
        batch["description"],
        padding="max_length",
        truncation=True,
        max_length=512
    )
    encodings["labels"] = np.array(batch["label_vec"], dtype=np.float32)
    return encodings

ds_train = (
    Dataset
    .from_pandas(train_df[["description", "label_vec"]])
    .map(tokenize_batch, batched=True, remove_columns=["description", "label_vec"])
    .with_format("torch")
)

ds_val = (
    Dataset
    .from_pandas(val_df[["description", "label_vec"]])
    .map(tokenize_batch, batched=True, remove_columns=["description", "label_vec"])
    .with_format("torch")
)
f1_metric = evaluate.load("f1", "multilabel")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs  = torch.sigmoid(torch.tensor(logits))
    y_pred = (probs >= 0.5).int().numpy()
    y_true = labels

    micro_f1 = f1_metric.compute(
        predictions=y_pred, references=y_true, average="micro"
    )["f1"]

    macro_f1 = f1_metric.compute(
        predictions=y_pred, references=y_true, average="macro"
    )["f1"]

    h_loss = hamming_loss(y_true, y_pred)

    print(f"\nMicro-averaged F1 : {micro_f1:.4f}")
    print(f"Macro-averaged F1 : {macro_f1:.4f}")
    print(f"Hamming Loss      : {h_loss:.4f}\n")
    print("Per-label report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=mlb.classes_,
            digits=2,
            zero_division=0
        )
    )

    return {
        "micro_f1":    micro_f1,
        "macro_f1":    macro_f1,
        "hamming_loss": h_loss,
    }

# Train
training_args = TrainingArguments(
    output_dir="distilbert-multilabel",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    num_train_epochs=5,
    weight_decay=0.01,
    metric_for_best_model="micro_f1",
    eval_strategy="epoch",
    logging_strategy="epoch",
    load_best_model_at_end=False,
    fp16=torch.cuda.is_available(),
    seed=SEED
)

class WeightedTrainer(Trainer):
    def __init__(self, pos_weight, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # move once to the right device later
        self.model.register_buffer("pos_weight", pos_weight)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        # labels = inputs["labels"]
        # outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs.logits, labels, pos_weight=model.pos_weight
        )
        return (loss, outputs) if return_outputs else loss

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=ds_train,
#     eval_dataset=ds_val,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics,
# )

trainer = WeightedTrainer(
    pos_weight=pos_weight,
    model=model,
    args=training_args,
    train_dataset=ds_train,
    eval_dataset=ds_val,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train(
    # resume_from_checkpoint=True
)

trainer.save_model("distilbert-multilabel/best")
tokenizer.save_pretrained("distilbert-multilabel/best")
mlb_path = "distilbert-multilabel/label_binarizer.pkl"
pd.to_pickle(mlb, mlb_path)

classifier = pipeline(
    "text-classification",
    model="distilbert-multilabel/best",
    tokenizer="distilbert-multilabel/best",
    top_k=None,
    function_to_apply="sigmoid"
)
mlb = pickle.load(open(mlb_path, "rb"))

# Micro-averaged F1 : 0.4052
# Macro-averaged F1 : 0.2676
# Hamming Loss      : 0.0841
#
# Per-label report:
#                      precision    recall  f1-score   support
#
#       binary search       0.00      0.00      0.00       136
#    bit manipulation       0.75      0.20      0.32        75
#       combinatorics       0.58      0.22      0.31        65
#     data structures       0.54      0.20      0.29       239
#  divide and conquer       0.00      0.00      0.00        32
# dynamic programming       0.65      0.12      0.20       257
#         game theory       1.00      0.10      0.19        29
#            geometry       0.69      0.30      0.42        37
#              graphs       0.75      0.50      0.60       234
#              greedy       0.60      0.39      0.48       323
#             hashing       0.00      0.00      0.00        29
#         interactive       0.00      0.00      0.00        20
#                math       0.66      0.41      0.51       348
#            matrices       0.91      0.57      0.70        37
#       number theory       0.73      0.30      0.43        90
#       probabilities       1.00      0.04      0.07        28
#       shortest path       0.00      0.00      0.00        34
#             sorting       0.00      0.00      0.00       149
#             strings       0.75      0.81      0.78       140
#               trees       0.87      0.46      0.60       126
#        two pointers       0.00      0.00      0.00        74
#          union find       0.00      0.00      0.00        36
#
#           micro avg       0.69      0.29      0.41      2538
#           macro avg       0.48      0.21      0.27      2538
#        weighted avg       0.55      0.29      0.36      2538
#         samples avg       0.50      0.31      0.36      2538