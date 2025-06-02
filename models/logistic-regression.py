import pandas as pd
import ast

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, classification_report, hamming_loss
from sklearn.svm import LinearSVC

from data_manager.utils import get_dataset_filepath

# Prepare Dataset
problems_df = pd.read_csv(get_dataset_filepath("problems.csv"))
problems_df['labels'] = problems_df['labels'].apply(ast.literal_eval)

# Vectorize
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    analyzer='word',
    max_df=0.9,
    min_df=5,
    stop_words='english'
)
X = vectorizer.fit_transform(problems_df['description'])

# Binarize labels
mlb = MultiLabelBinarizer()
Y   = mlb.fit_transform(problems_df['labels'])

# Train
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.20, random_state=42
)

clf = OneVsRestClassifier(LogisticRegression(
    solver='liblinear',
    max_iter=1_000,
    class_weight='balanced',
    C=3.0
))

clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)

print(f"Micro-averaged F1 : {f1_score(y_test, y_pred, average='micro'):.4f}")
print(f"Macro-averaged F1 : {f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"Hamming Loss: {hamming_loss(y_test, y_pred):.4f}\n") # todo: what is it

print("Per-label report:")
print(classification_report(y_test, y_pred, target_names=mlb.classes_))

def predict_labels(description: str, threshold: float = 0.5):
    X_new = vectorizer.transform([description])
    proba = clf.predict_proba(X_new)[0]
    idx   = (proba >= threshold)
    return mlb.classes_[idx].tolist()
