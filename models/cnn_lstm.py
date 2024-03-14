import torch
from torch import nn


class cnn_lstm(nn.Module):

    def __init__(self, in_channels=22, dims=[32, 64, 128, 256], num_classes=4, kernel_size=7, stride=1, pad=3, dropout=0.5, num_layers=1, hidden_size=512):
        super().__init__()

        self.hidden_size = hidden_size
        self.fc_in = hidden_size

        self.conv_modules = nn.ModuleList()
        prev_channels = in_channels
        for dim in dims:
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels=prev_channels, out_channels=dim, kernel_size=kernel_size, stride=stride, padding=pad),
                nn.ELU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.BatchNorm1d(num_features=dim),
                nn.Dropout(dropout),
            )
            self.conv_modules.append(conv_block)
            prev_channels = dim

        self.lstm = nn.LSTM(input_size=dims[-1], hidden_size=self.hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(in_features=self.fc_in, out_features=num_classes)

    def forward(self, x: torch.Tensor):
        for conv_module in self.conv_modules:
            x = conv_module(x)
        # x: B, C, L
        x = x.permute(0, 2, 1)
        # x: B, L, C
        x, (h_n, c_n) = self.lstm(x)
        # x: B, L, hidden_size
        out = nn.functional.relu(x[:, -1])
        # out: B, hidden_size
        out = self.fc(out)
        # out: B, num_classes
        return out
    
    def compute_l1_loss(self):
        l1_loss = 0
        for param in self.parameters():
            l1_loss += torch.abs(param).sum()
        return l1_loss