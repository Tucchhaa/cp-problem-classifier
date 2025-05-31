#
# This code maps dataset from huggingface to desired format for training
#

import pandas as pd
import ast

from utils import get_dataset_filepath, convert_codeforces_labels, convert_leetcode_labels

MAX_PROBLEM_DESCRIPTION_LENGTH = 6000
MAX_LABELS_COUNT = 7

class Formatter:
    def __init__(self, dataset_filepath):
        self.dataset_filepath = dataset_filepath

    def format(self):
        loaded_df = pd.read_csv(self.dataset_filepath)
        loaded_df['labels'] = loaded_df['labels'].apply(ast.literal_eval)

        loaded_df = loaded_df.apply(self._format_row, axis=1)
        loaded_df = loaded_df.dropna()

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
            if not pd.isna(row[field]):
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
        super().__init__("scrapper/spoj.json")

    def _format_row(self, row):
        pass

    def _get_description(self, row):
        pass

