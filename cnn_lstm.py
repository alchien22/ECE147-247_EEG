import torch
from torch import nn


class cnn_lstm(nn.Module):

    def __init__(self, in_channels=22, conv_blocks=3, dims=[32, 64, 128], num_classes=4, kernel_size=7, stride=1, pad=3, dropout=0.5):
        super().__init__()

        self.conv_modules = nn.ModuleList()
        prev_channels = in_channels
        for i in range(conv_blocks):
            out_channels = dims[i]
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels=prev_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=pad),
                nn.ELU(),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
                nn.BatchNorm1d(num_features=out_channels),
                nn.Dropout(dropout),
            )
            self.conv_modules.append(conv_block)
            prev_channels = out_channels
        
        conv_out_size = dims[-1]
        self.fc1 = nn.Linear(in_features=conv_out_size, out_features=256)
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=1, batch_first=True)
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, x):
        for conv_module in self.conv_modules:
            x = conv_module(x)
        # x: B, C, L
        x = x.permute(0, 2, 1)
        # x: B, L, C
        x = self.fc1(x)
        # x: B, L, C
        x = nn.functional.elu(x)
        x, _ = self.lstm(x)
        # h_n: num_layers, B, hidden_size
        # x: B, L, hidden_size
        out = x[:, -1]
        # out: B, hidden_size
        out = self.fc2(out)
        # out: B, num_classes
        return out
    
    def compute_l1_loss(self):
        l1_loss = 0
        for param in self.parameters():
            l1_loss += torch.abs(param).sum()
        return l1_loss