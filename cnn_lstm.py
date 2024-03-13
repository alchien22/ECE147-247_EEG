import torch
from torch import nn


class cnn_lstm(nn.Module):

    def __init__(self, in_channels=22, in_length=400, conv_blocks=3, dims=[64, 128, 256], num_classes=4, kernel_size=7, stride=1, pad=3, dropout=0.5):
        super().__init__()
        
        self.conv_modules = nn.ModuleList()
        prev_channels = in_channels
        for i in range(conv_blocks):
            out_channels = dims[i]
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels=prev_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=pad),
                nn.LeakyReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
                nn.BatchNorm1d(num_features=out_channels),
                nn.Dropout(dropout),
            )
            self.conv_modules.append(conv_block)
            prev_channels = out_channels
        
        # 400 -> 200 -> 100  
        flatten_size = dims[-1] * (in_length // 2**conv_blocks)
        self.fc1 = nn.Linear(in_features=flatten_size, out_features=256)
        self.activation1 = nn.LeakyReLU()
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True, dropout=0.4)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

# (conv -> max -> batchnorm -> dropout) x 4
# flatten -> FC -> LSTM -> FC -> softmax

    def forward(self, x):
        for conv_module in self.conv_modules:
            x = conv_module(x)
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        out = self.activation1(out)
        out = out.unsqueeze(1)
        out, (h_n, n_n) = self.lstm(out)
        # Just use last hidden state
        out = self.fc2(h_n.squeeze(0))
        return out
    
    def compute_l1_loss(self):
        l1_loss = 0
        for param in self.parameters():
            l1_loss += torch.abs(param).sum()
        return l1_loss