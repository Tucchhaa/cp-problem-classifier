import os

import pandas as pd

from data_manager.load import OpenR1CodeforcesLoader, KaysssLeetcodeLoader
from data_manager.format import OpenR1CodeforcesFormatter, KaysssLeetcodeFormatter, SpojFormatter
from data_manager.spoj_scrapper.scrapper import Scrapper
from data_manager.plot import plot_figures
from data_manager.utils import get_dataset_filepath

absolute_path = os.path.dirname(os.path.abspath(__file__))

# === Loading state ====
print("PREPARE DATASET: Dataset loading stage")
codeforcesLoader = OpenR1CodeforcesLoader()
leetcodeLoader = KaysssLeetcodeLoader()

# set force=True, to redownload datasets
codeforcesLoader.download(force=False)
leetcodeLoader.download(force=False)

# === Scrapper stage ===
print("PREPARE DATASET: SPOJ problems scrapping stage")

scrapper = Scrapper()
scrapper.start()

# === Format stage ====
print("PREPARE DATASET: Dataset formatting stage")
codeforcesFormatter = OpenR1CodeforcesFormatter()
leetcodeFormatter = KaysssLeetcodeFormatter()
spojFormatter = SpojFormatter()

problems_df = pd.concat([
    codeforcesFormatter.format(),
    leetcodeFormatter.format(),
    spojFormatter.format()
])

dataset_filepath = get_dataset_filepath('problems.csv')

if os.path.exists(dataset_filepath):
    os.remove(dataset_filepath)

problems_df.to_csv(dataset_filepath, index=False)

# === Plot stage ===
print("PREPARE DATASET: Plotting stage")

plot_figures()

