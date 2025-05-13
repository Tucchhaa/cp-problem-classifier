import ast
import matplotlib.pyplot as plt
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from typing import List, Tuple
import pandas as pd

def load_dataset(file='dataset/problems.csv') -> Tuple[List[str], List[List[str]]]:
    """
    load dataset and tokenize input and label
    """
    df = pd.read_csv(file)

    # Tokenize labels
    # (TODO) The number of labels is fixed, we can write a table to match label and its token)
    labels = [ast.literal_eval(l) for l in df['labels']]
    all_labels = set(b for a in labels for b in a)
    label_dict = {j: i for i, j in enumerate(all_labels)}
    label_num = len(label_dict)
    label_vecs = []
    for ls in labels:
        lv = [0] * label_num
        for l in ls:
            lv[label_dict[l]] = 1
        label_vecs.append(lv)

    # Tokenize descriptions
    descriptions = []
    voc_dict = {'<pad>': 0, "<unk>": 1}
    voc_count = 2
    for description in df['description']:
        desc = []
        for word in description.split():
            if not word in voc_dict:
                voc_dict[word] = voc_count
                voc_count += 1
            desc.append(voc_dict[word])
        descriptions.append(desc)

    return descriptions, label_vecs, voc_dict, label_dict

class TrainDataset(Dataset):
    def __init__(self, descriptions, labels):
        self.descs = descriptions
        self.labels = labels

    def __len__(self):
        return len(self.descs)

    def __getitem__(self, idx):
        desc = torch.tensor(self.descs[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return desc, label

def collate_fn(batch):
    """
    Padding data in same batch into same length
    """
    descs, labels = zip(*batch)

    padded_descs = pad_sequence(descs, batch_first=True, padding_value=0)

    labels = torch.stack(labels)
    return padded_descs, labels

def plotloss(train_losses, val_losses):

    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label='Training Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss.png')
    plt.close()
