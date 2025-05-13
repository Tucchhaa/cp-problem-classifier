#
# This code maps dataset from huggingface to desired format for training
#

import pandas as pd
import os
import ast
from bs4 import BeautifulSoup  
from utils import get_dataset_filepath, convert_codeforces_labels, convert_leetcode_labels

MAX_PROBLEM_DESCRIPTION_LENGTH = 6000
MAX_LABELS_COUNT = 7

class Formatter:
    def __init__(self, dataset_name, source):
        self.dataset_name = dataset_name
        self.source = source

    def format(self):
        dataset_filepath = get_dataset_filepath(self.dataset_name, self.source)

        loaded_df = pd.read_csv(dataset_filepath)
        loaded_df['labels'] = loaded_df['labels'].apply(ast.literal_eval)

        loaded_df = loaded_df.apply(self._format_row, axis=1)
        loaded_df = loaded_df.dropna()

        return loaded_df

    def _format_row(self, row):
        raise NotImplementedError()


class OpenR1CodeforcesFormatter(Formatter):
    def __init__(self):
        super().__init__("open-r1/codeforces", "huggingface")

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
        super().__init__("kaysss/leetcode-problem-detailed", "huggingface")

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
        
        if pd.isna(row['description']):
            return None

        # Use BeautifulSoup to clean HTML tags
        raw_description = row['description'].replace('\n', ' ')
        clean_description = BeautifulSoup(raw_description, "html.parser").get_text()
        # remove extra spaces
        result = " ".join(clean_description.split())           
        return result

# TODO:
class SpojFormatter(Formatter):
    def __init__(self):
        super().__init__("kaysss/leetcode-problem-detailed", "scrapper")

    def _format_row(self, row):
        pass

    def _get_description(self, row):
        pass


codeforcesFormatter = OpenR1CodeforcesFormatter()
leetcodeFormatter = KaysssLeetcodeFormatter()

dataset_df = pd.concat([
    codeforcesFormatter.format(),
    leetcodeFormatter.format()
])

dataset_filepath = get_dataset_filepath('problems', 'huggingface','jsonl')

if os.path.exists(dataset_filepath):
    os.remove(dataset_filepath)

# dataset_df.to_csv(dataset_filepath, index=False)
dataset_df.to_json(dataset_filepath, orient='records', lines=True)

