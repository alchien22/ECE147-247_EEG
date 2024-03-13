import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import List
from preprocessing import load_data
from utils import train, evaluate

train_dataloader, val_dataloader, test_dataloader = load_data(batch_size=32)

class LSTM(nn.Module):
    def __init__(self, input_dim: int, conv_dims: List[int], hidden_dim: int, num_layers: int):
        super(LSTM, self).__init__()
        
        self.conv = nn.ModuleList()
        self.input_dim = input_dim
        self.conv_dims = conv_dims
        self.hidden_dim = hidden_dim

        prev_dim = input_dim
        for dim in conv_dims:
            self.conv.append(nn.Conv1d(prev_dim, dim, kernel_size=7, padding=3))
            self.conv.append(nn.ELU())
            self.conv.append(nn.MaxPool1d(kernel_size=2, stride=2))
            self.conv.append(nn.BatchNorm1d(dim))
            self.conv.append(nn.Dropout(0.5))
            prev_dim = dim
        
        self.lstm = nn.LSTM(prev_dim, hidden_dim, num_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, 4)
        
    def forward(self, x):
        for layer in self.conv:
            x = layer(x)
        x = torch.transpose(x, 1, 2)
        out, (h_n, c_n) = self.lstm(x)
        out = self.out(F.relu(out[:, -1]))
        return out

model = LSTM(input_dim=22, conv_dims=[32, 64], hidden_dim=128, num_layers=1)
model.to('cpu')

optimizer = Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

num_epochs = 100
for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_dataloader, optimizer, criterion, torch.device('cpu'))
    val_loss, val_acc = evaluate(model, val_dataloader, criterion, torch.device('cpu'))
    print(f"Epoch {epoch + 1}/{num_epochs} loss: {train_loss}, acc: {train_acc}, val_loss: {val_loss}, val_acc: {val_acc}")