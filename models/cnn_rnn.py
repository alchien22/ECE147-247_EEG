import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import List
from utils.utils import fit

class RNN(nn.Module):
    def __init__(self, input_dim: int, conv_dims: List[int], hidden_dim: int, num_layers: int):
        super(RNN, self).__init__()
        
        self.conv = nn.ModuleList()
        self.input_dim = input_dim
        self.conv_dims = conv_dims
        self.hidden_dim = hidden_dim

        prev_dim = input_dim
        for dim in conv_dims:
            self.conv.append(nn.Conv1d(prev_dim, dim, kernel_size=5, padding=2))
            self.conv.append(nn.ELU())
            self.conv.append(nn.MaxPool1d(kernel_size=2, stride=2))
            self.conv.append(nn.BatchNorm1d(dim))
            self.conv.append(nn.Dropout1d())
            prev_dim = dim
        
        self.rnn = nn.RNN(prev_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.out = nn.Linear(hidden_dim, 4)
        
    def forward(self, x):
        for layer in self.conv:
            x = layer(x)
        x = torch.transpose(x, 1, 2)
        out, h_n = self.rnn(x)
        out = self.out(F.relu(out[:, -1]))
        return out


if __name__ == "__main__":
    device = torch.device('cpu')

    model = RNN(input_dim=22, conv_dims=[32, 64, 128, 256], hidden_dim=128, num_layers=1)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    fit(model, optimizer, criterion, num_epochs=100, device=device)