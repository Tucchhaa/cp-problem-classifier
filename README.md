# Competitive programming problems classifier

## Commands
- `python ./data_manager/load.py` - downloads datasets from hugging face. Datasets will be downloaded to `data_manager/datasets/huggingface`
- `python ./data_manager/spoj_scrapper/scrapper.py` - scraps data from SPOJ website and saves it in `./data_manager/dataset/scrapper/spoj.json`
- `python ./data_manager/format.py` - creates dataset for training the model. It will produce `data_manager/dataset/problems.csv` file
- `python ./data_manager/plot.py` - plots charts according to `data_manager/dataset/problems.csv` file. Charts are produced in `data_manager/figures` directory

## Dataset

**Datasets used**
- codeforces: https://huggingface.co/datasets/open-r1/codeforces
- leetcode: https://huggingface.co/datasets/kaysss/leetcode-problem-detailed
