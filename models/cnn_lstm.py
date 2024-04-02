import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import List
from utils.utils import fit

class LSTM(nn.Module):
    def __init__(self, input_dim: int, conv_dims: List[int], hidden_dim: int, num_layers: int, in_length=400):
        super(LSTM, self).__init__()
        
        self.conv = nn.ModuleList()
        self.input_dim = input_dim
        self.conv_dims = conv_dims
        self.hidden_dim = hidden_dim

        prev_dim = input_dim
        for i, dim in enumerate(conv_dims):
            self.conv.append(nn.Conv1d(prev_dim, dim, kernel_size=11, padding=5))
            # self.conv.append(nn.ELU())
            self.conv.append(nn.ReLU())
            if i == 1:
                self.conv.append(nn.MaxPool1d(kernel_size=2, stride=2))
            self.conv.append(nn.BatchNorm1d(dim))
            self.conv.append(nn.Dropout(0.4))
            prev_dim = dim
        
        self.lstm = nn.LSTM(prev_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False)
        self.out = nn.Linear(hidden_dim * (in_length // 2), 4)
        
    def forward(self, x):
        for layer in self.conv:
            x = layer(x)
        x = torch.transpose(x, 1, 2)
        # out, (h_n, c_n) = self.lstm(x)
        # out = self.out(F.relu(out[:, -1]))
        out, (h_n, c_n) = self.lstm(x)
        out = self.out(torch.flatten(out, 1))
        return out


if __name__ == '__main__':
    device = torch.device('mps')

    model = LSTM(input_dim=22, conv_dims=[16, 32, 48, 64], hidden_dim=64, num_layers=1)
    # model = LSTM(input_dim=22, conv_dims=[32, 64], hidden_dim=128, num_layers=1)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    fit(model, optimizer, criterion, num_epochs=2, device=device)