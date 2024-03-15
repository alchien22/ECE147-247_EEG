import torch
import torch.nn as nn
from utils.seed import *
from utils.preprocessing import *
from utils.loops import *
import matplotlib.pyplot as plt
from utils.utils import fit

class CNN(nn.Module):
    def __init__(self, in_channels=22, in_length=400, conv_blocks=3, dims=[64, 128, 256], num_classes=4, kernel_size=11, stride=1, pad=5, dropout=0.5):
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
        self.fc1 = nn.Linear(in_features=flatten_size, out_features=128)
        self.activation1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.activation2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        for conv_module in self.conv_modules:
            x = conv_module(x)
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        out = self.activation1(out)
        out = self.fc2(out)
        out = self.activation2(out)
        out = self.fc3(out)
        return out
    
    def compute_l1_loss(self):
        l1_loss = 0
        for param in self.parameters():
            l1_loss += torch.abs(param).sum()
        return l1_loss
    

if __name__ == '__main__':
    device = torch.device('cpu')
    # (H - h + 2p) / s + 1
    # 3 -> 1, 5 -> 2, 7 -> 3, 11 -> 5, 13 -> 6
    kernel_size = 11
    pad = 5

    model = CNN(kernel_size=kernel_size, pad=pad)
    model.to(device)

    weight_decay = 1e-2

    lr = 1e-3
    optimizer = torch.optim.SGD(params=model.parameters(), momentum=0.9, lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    fit(model, optimizer, criterion, num_epochs=62, device=device)
