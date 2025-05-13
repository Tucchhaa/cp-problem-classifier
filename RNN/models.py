import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from typing import List

class RNN(nn.Module):
    def __init__(self, voc_size, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        self.embedding = nn.Embedding(voc_size, input_size)
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hn = self.rnn(embedded)
        output = self.fc(output[:,-1,:])
        return output
    
class LSTM(nn.Module):
    def __init__(self, voc_size, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        self.embedding = nn.Embedding(voc_size, input_size)
        
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hn, cn) = self.rnn(embedded)
        output = self.fc(output[:,-1,:])
        return output

def train(model, loader: DataLoader, criterion, optimizer, device) -> float:

    model.train()
    total_loss = 0.0

    for inputs, labels in tqdm(loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Return average loss
    return total_loss / len(loader)

def validate(model, loader: DataLoader, criterion, device, threshold) -> float:

    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            probs = torch.sigmoid(outputs)
            all_preds.append((probs.cpu() > threshold).int())
            all_labels.append(labels)

    all_preds = torch.cat(all_preds, dim=0).cpu().numpy()
    all_labels = torch.cat(all_labels, dim=0).cpu().numpy()

    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    subset_acc = accuracy_score(all_labels, all_preds)

    return total_loss / len(loader), subset_acc, f1_micro

def predict(model: RNN, input, threshold) -> List[List[int]]:

    model.eval()
    preds = []
    with torch.no_grad():
        outputs = model(input)
        probs = torch.sigmoid(outputs)
        preds.append((probs.cpu() > threshold).int())

    return preds
