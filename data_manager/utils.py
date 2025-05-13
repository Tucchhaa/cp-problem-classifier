import os

from problem_types import codeforces_to_standard, leetcode_to_standard
from typing import List, Literal

Source = Literal["huggingface", "scrapper", ""]

def get_dataset_filepath(dataset_name: str, source: Source = "") -> str:
    dest_dir = f"./dataset/{source}/"
    extension = '.json' if source == 'scrapper' else '.csv'

    filename = dataset_name.replace('/', '_')
    os.makedirs(dest_dir, exist_ok=True)

    return f"{dest_dir}{filename}{extension}"

def _convert_labels(labels: List[str], labels_map) -> List[str]:
    converted_labels = []

    for label in labels:
        if label not in labels_map:
            raise RuntimeError()

        mapped_labels = labels_map[label]

        if mapped_labels is None:
            return []

        converted_labels.extend(mapped_labels)

    return converted_labels

def convert_codeforces_labels(labels: List[str]) -> List[str]:
    return _convert_labels(labels, codeforces_to_standard)

def convert_leetcode_labels(labels: List[str]) -> List[str]:
    return _convert_labels(labels, leetcode_to_standard)

