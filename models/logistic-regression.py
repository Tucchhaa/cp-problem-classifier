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
# clf = OneVsRestClassifier(LinearSVC(class_weight='balanced'))

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


# Micro-averaged F1 : 0.5292
# Macro-averaged F1 : 0.5048
# Hamming Loss: 0.1024
#
# Per-label report:
#                      precision    recall  f1-score   support
#
#       binary search       0.31      0.42      0.36       294
#    bit manipulation       0.39      0.40      0.40       143
#       combinatorics       0.51      0.59      0.55       142
#     data structures       0.50      0.62      0.56       519
#  divide and conquer       0.13      0.12      0.12        60
# dynamic programming       0.39      0.47      0.43       516
#         game theory       0.55      0.80      0.65        51
#            geometry       0.47      0.75      0.58        85
#              graphs       0.62      0.67      0.64       429
#              greedy       0.54      0.63      0.58       638
#             hashing       0.18      0.14      0.16        50
#         interactive       0.98      0.94      0.96        49
#                math       0.58      0.65      0.61       687
#            matrices       0.68      0.68      0.68        78
#       number theory       0.55      0.58      0.56       168
#       probabilities       0.60      0.69      0.64        48
#       shortest path       0.32      0.46      0.38        69
#             sorting       0.33      0.46      0.39       299
#             strings       0.74      0.80      0.77       279
#               trees       0.65      0.70      0.68       235
#        two pointers       0.15      0.22      0.18       140
#          union find       0.20      0.31      0.25        70
#
#           micro avg       0.49      0.58      0.53      5049
#           macro avg       0.47      0.55      0.50      5049
#        weighted avg       0.50      0.58      0.53      5049
#         samples avg       0.49      0.60      0.50      5049