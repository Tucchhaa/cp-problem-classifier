import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report
from typing import List, Tuple

class RNN(nn.Module):
    def __init__(self, voc_size, input_size, hidden_size, output_size, layers, model='RNN'):
        super(RNN, self).__init__()

        self.modelname = model
        
        self.embedding = nn.Embedding(voc_size, input_size)
        if model == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, layers, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden_size * 2, output_size)
        elif model == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, layers, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden_size * 2, output_size)
        else:
            self.rnn = nn.RNN(input_size, hidden_size, layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        self.dropout = nn.Dropout(p=0.3)

        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.rnn(x)
        output = self.dropout(output[:,-1,:])
        output = self.fc(output)
        return output

def train(model, loader: DataLoader, criterion, optimizer, device) -> float:

    model.train()
    total_loss = 0.0

    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Return average loss
    return total_loss / len(loader)

def validate(model, loader: DataLoader, criterion, device, print_report=False):

    model.eval()
    

    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            probs = torch.sigmoid(outputs)
            all_preds.append(probs)
            all_labels.append(labels)
    
    all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
    all_preds = torch.cat(all_preds, dim=0).cpu().numpy()

    res_thres = 0
    res_f1 = 0
    res_preds = []
    for t in (i/100 for i in range(20, 80, 5)):

        f1 = float(f1_score(all_labels, all_preds, average='macro', zero_division=0))
        # subset_acc = float(accuracy_score(all_labels, all_preds))

        if f1 > res_f1:
            res_thres = t
            res_f1 = f1
            res_preds = all_preds

    if print_report:
        print(classification_report(all_labels, res_preds, zero_division=0))

    return total_loss / len(loader), res_f1, res_thres

def predict(model: RNN, input, threshold) -> List[List[int]]:

    model.eval()
    preds = []
    with torch.no_grad():
        outputs = model(input)
        probs = torch.sigmoid(outputs)
        preds.append((probs.cpu() > threshold).int())

    return preds
