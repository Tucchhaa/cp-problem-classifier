#
# This code maps dataset from huggingface to desired format for training
#
from pathlib import Path

import pandas as pd
import ast

from utils import get_dataset_filepath, convert_codeforces_labels, convert_leetcode_labels, convert_spoj_labels

MAX_PROBLEM_DESCRIPTION_LENGTH = 6000
MAX_LABELS_COUNT = 7

class Formatter:
    def __init__(self, dataset_filepath):
        self.dataset_filepath = dataset_filepath

    def format(self):
        extension = Path(self.dataset_filepath).suffix

        if extension == '.csv':
            loaded_df = pd.read_csv(self.dataset_filepath)
        else:
            loaded_df = pd.read_json(self.dataset_filepath)

        if 'labels' in loaded_df.columns:
            loaded_df['labels'] = loaded_df['labels'].apply(ast.literal_eval)

        loaded_df = loaded_df.apply(self._format_row, axis=1)
        loaded_df = loaded_df.dropna()

        if isinstance(loaded_df, pd.Series):
            loaded_df = pd.DataFrame(loaded_df.tolist()).reset_index(drop=True)

        return loaded_df

    def _format_row(self, row):
        raise NotImplementedError()


class OpenR1CodeforcesFormatter(Formatter):
    def __init__(self):
        dataset_filepath = get_dataset_filepath(f"huggingface/open-r1_codeforces.csv")
        super().__init__(dataset_filepath)

    def _format_row(self, row):
        labels = convert_codeforces_labels(row['labels'])

        if len(labels) == 0 or len(labels) > MAX_LABELS_COUNT:
            return None

        description = self._get_description(row)

        if not description or len(description) > MAX_PROBLEM_DESCRIPTION_LENGTH:
            return None

        return pd.Series({
            'source': 'codeforces',
            'title': row['title'],
            'description': description,
            'labels': labels,
        })

    def _get_description(self, row):
        if pd.isna(row['description']):
            return None

        result = row['description'].replace('\n', ' ')
        fields = ['input_format', 'output_format', 'interaction_format', 'note']

        for field in fields:
            if not pd.isna(row[field]) and len(row[field]) > 0:
                text = row[field].replace('\n', ' ')
                result += f"\n{field} = {text}"

        return result


class KaysssLeetcodeFormatter(Formatter):
    def __init__(self):
        dataset_filepath = get_dataset_filepath(f"huggingface/kaysss_leetcode-problem-detailed.csv")
        super().__init__(dataset_filepath)

    def _format_row(self, row):
        labels = convert_leetcode_labels(row['labels'])

        if len(labels) == 0 or len(labels) > MAX_LABELS_COUNT:
            return None

        description = self._get_description(row)

        if not description or len(description) > MAX_PROBLEM_DESCRIPTION_LENGTH:
            return None

        return pd.Series({
            'source': 'leetcode',
            'title': row['title'],
            'description': description,
            'labels': labels,
        })

    def _get_description(self, row):
        #
        # TODO: parse html, and split data to description, input_format, output_format, constraints,
        # TODO: scrap hints from the website
        #

        if pd.isna(row['description']):
            return None

        result = row['description'].replace('\n', ' ')

        return result

# TODO:
class SpojFormatter(Formatter):
    def __init__(self):
        dataset_filepath = get_dataset_filepath(f"scrapper/spoj.json")
        super().__init__(dataset_filepath)

    def _format_row(self, row):
        if row.user_count < 20:
            return None

        labels = convert_spoj_labels(row['tags'])

        if len(labels) == 0 or len(labels) > MAX_LABELS_COUNT:
            return None

        description = self._get_description(row)

        if not description or len(description) > MAX_PROBLEM_DESCRIPTION_LENGTH:
            return None

        return pd.Series({
            'source': 'spoj',
            'title': row['title'],
            'description': description,
            'labels': labels,
        })

    def _get_description(self, row):
        if pd.isna(row['description']):
            return None

        result = row['description'].replace('\n', ' ')
        fields = ['task_description', 'input_format', 'output_format']

        for field in fields:
            if not pd.isna(row[field]) and len(row[field]) > 0:
                text = row[field].replace('\n', ' ')
                text = self.safe_fix_mojibake(text)
                result += f"\n{field} = {text}"

        return result

    def safe_fix_mojibake(self, s: str):
        try:
            return s.encode('latin1').decode('utf-8')
        except:
            return s
