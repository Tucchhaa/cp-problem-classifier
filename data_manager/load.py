#
# This code downloads datasets from huggingface and saves only needed columns
#

from datasets import load_dataset
import pandas as pd
import os
import numpy as np
from utils import get_dataset_filepath

class DatasetLoader:
    def __init__(self, dataset_name, dataset_filepath):
        self.dataset_name = dataset_name
        self.dataset_filepath = dataset_filepath

    def download(self, force=False):
        if os.path.exists(self.dataset_filepath):
            if force:
                os.remove(self.dataset_filepath)
            else:
                return

        dataset_df = self._load()
        dataset_df = self._map(dataset_df)
        dataset_df.to_csv(self.dataset_filepath, index=False)

    def _load(self):
        dataset = load_dataset(self.dataset_name)

        train_data = dataset["train"]
        train_data_df = train_data.to_pandas()

        if 'test' in dataset:
            test_data = dataset["test"]
            test_data_df = test_data.to_pandas()

            combined_df = pd.concat([train_data_df, test_data_df], ignore_index=True)

            return combined_df

        return train_data_df

    def _map(self, loaded_df):
        result_df = loaded_df.apply(self._map_row, axis=1)

        return result_df

    def _map_row(self, row):
        raise NotImplementedError()


class OpenR1CodeforcesLoader(DatasetLoader):
    def __init__(self):
        super().__init__(
            "open-r1/codeforces",
            get_dataset_filepath("huggingface/open-r1_codeforces.csv")
        )

    def _load(self):
        df = super()._load()

        # convert to list, so that tags will be saved in list format
        df["tags"] = df["tags"].apply(lambda x: list(x) if isinstance(x, np.ndarray) else x)

        return df

    def _map_row(self, row):
        return pd.Series({
            'id': row['id'],
            'title': row['title'],
            'labels': row['tags'],
            'time_limit_per_test': row['time_limit'],
            'memory_limit_per_test': row['memory_limit'],
            'description': row['description'],
            'input_format': row['input_format'],
            'output_format': row['output_format'],
            'interaction_format': row['interaction_format'],
            'note': row['note'],
            'examples': row['examples'],
        })


class KaysssLeetcodeLoader(DatasetLoader):
    def __init__(self):
        super().__init__(
            "kaysss/leetcode-problem-detailed",
            get_dataset_filepath("huggingface/kaysss_leetcode-problem-detailed.csv")
        )

    def _map_row(self, row):
        return pd.Series({
            'id': row['questionFrontendId'],
            'title': row['questionTitle'],
            'titleKebabCase': row['TitleSlug'],
            'labels': row['topicTags'],
            'difficulty': row['difficulty'],
            'description': row['content']
        })
