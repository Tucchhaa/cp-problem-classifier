import os

from data_manager.problem_types import codeforces_to_standard, leetcode_to_standard
from typing import List, Literal

Source = Literal["huggingface", "scrapper", ""]

def get_dataset_filepath(dataset_filename: str) -> str:
    absolute_path = os.path.dirname(os.path.abspath(__file__))
    filepath = f"{absolute_path}/dataset/{dataset_filename}"

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    return filepath

def _convert_labels(labels: List[str], labels_map) -> List[str]:
    converted_labels = []

    for label in labels:
        if label not in labels_map:
            raise RuntimeError()

        mapped_labels = labels_map[label]

        if mapped_labels is None:
            return []

        converted_labels.extend(mapped_labels)

    return list(set(converted_labels))

def convert_codeforces_labels(labels: List[str]) -> List[str]:
    return _convert_labels(labels, codeforces_to_standard)

def convert_leetcode_labels(labels: List[str]) -> List[str]:
    return _convert_labels(labels, leetcode_to_standard)
