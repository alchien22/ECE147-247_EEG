import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import List
from utils.preprocessing import load_data
from utils.utils import train, evaluate

train_dataloader, val_dataloader, test_dataloader = load_data(batch_size=32)

class LSTM(nn.Module):
    def __init__(self, input_dim: int, conv_dims: List[int], hidden_dim: int, num_layers: int):
        super(LSTM, self).__init__()
        
        self.conv = nn.ModuleList()
        self.input_dim = input_dim
        self.conv_dims = conv_dims
        self.hidden_dim = hidden_dim

        prev_dim = input_dim
        for i, dim in enumerate(conv_dims):
            self.conv.append(nn.Conv1d(prev_dim, dim, kernel_size=5, padding=2))
            # self.conv.append(nn.ELU())
            self.conv.append(nn.ReLU())
            if i == 1:
                self.conv.append(nn.MaxPool1d(kernel_size=2, stride=2))
            self.conv.append(nn.BatchNorm1d(dim))
            self.conv.append(nn.Dropout(0.4))
            prev_dim = dim
        
        self.lstm = nn.LSTM(prev_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.out = nn.Linear(hidden_dim * 200 * 2, 4)
        
    def forward(self, x):
        for layer in self.conv:
            x = layer(x)
        x = torch.transpose(x, 1, 2)
        # out, (h_n, c_n) = self.lstm(x)
        # out = self.out(F.relu(out[:, -1]))
        out, (h_n, c_n) = self.lstm(x)
        out = self.out(torch.flatten(out, 1))
        return out

model = LSTM(input_dim=22, conv_dims=[16, 32, 48, 64], hidden_dim=64, num_layers=1)
# model = LSTM(input_dim=22, conv_dims=[32, 64], hidden_dim=128, num_layers=1)
model.to('mps')

optimizer = Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

num_epochs = 100
for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_dataloader, optimizer, criterion, torch.device('mps'))
    val_loss, val_acc = evaluate(model, val_dataloader, criterion, torch.device('mps'))
    print(f"Epoch {epoch + 1}/{num_epochs} loss: {train_loss}, acc: {train_acc}, val_loss: {val_loss}, val_acc: {val_acc}")
    if (epoch + 1) % 5 == 0:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        }, f'lstm_epoch{epoch + 1}.pt')