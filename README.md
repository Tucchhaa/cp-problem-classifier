# Competitive programming problems classifier

This is a tool designed to classify competitive programming problems with relevant topic labels.
It takes problem descriptions as input and returns multiple labels that reflect the underlying concepts of the problem.

Currently, two models are supported:

- One-vs-Rest + Logistic Regression: F1 score = 0.5048
- DistilBERT: F1 score = 0.51

## Dataset

### Preparation

```python ./data_manager/prepare_dataset.py```

- Prepare dataset at `data_manager/dataset/problems.csv`.
- You can edit `prepare_dataset.py` to manipulate dataset preparing pipeline.

### Dataset Sources

#### Used Datasets

- codeforces: https://huggingface.co/datasets/open-r1/codeforces
- leetcode: https://huggingface.co/datasets/kaysss/leetcode-problem-detailed

#### Scrapper

- https://www.spoj.com/

## Train Model

### Requirements

Install dependencies:
```pip install -r requirements.txt```

### One Vs Rest + Logistic regression

```python ./models/logistic-regression.py```

### DistilBERT

open file and run all cells in `model/DistilBERT.ipynb`.

## Demo

Start web server: `python fast_api/app.py`.

Then use browser to navigate to `127.0.0.1:8000`.
