import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        kernel_size = 3
        self.ConvLayer1 = nn.Sequential(
            nn.Conv1d(22, 25, kernel_size), 
            nn.MaxPool1d(kernel_size=kernel_size, padding=1),
            nn.BatchNorm1d(num_features=25),
            nn.ELU(),
            nn.Dropout1d(0.5)
        )
        self.ConvLayer2 = nn.Sequential(
            nn.Conv1d(25, 50, kernel_size), 
            nn.MaxPool1d(kernel_size=kernel_size, padding=1),
            nn.BatchNorm1d(num_features=50),
            nn.ELU(),
            nn.Dropout1d(0.5)
        )
        self.ConvLayer3 = nn.Sequential(
            nn.Conv1d(50, 100, kernel_size),  
            nn.MaxPool1d(kernel_size=kernel_size, padding=1),
            nn.BatchNorm1d(num_features=100),
            nn.ELU(),
            nn.Dropout1d(0.5)
        )
        self.ConvLayer4 = nn.Sequential(
            nn.Conv1d(100, 200, kernel_size),  
            nn.MaxPool1d(kernel_size=kernel_size, padding=1),
            nn.BatchNorm1d(num_features=200),
            nn.ELU(),
            nn.Dropout1d(0.5)
        ) 
        
        #200*4
        self.fc = nn.Linear(200*4, out_features=4) 
        
    def forward(self, x):
        x = self.ConvLayer1(x)
        x = self.ConvLayer2(x)
        x = self.ConvLayer3(x)
        x = self.ConvLayer4(x)
        #print(x.shape)
        #after conv x is (32, 200, 3)
        x = x.view(-1, 200*4)  # Maintain bsz dimension
        x = F.softmax(self.fc(x), dim=1)
        return x
