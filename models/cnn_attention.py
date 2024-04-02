import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Attention_Model(nn.Module):
    def __init__(self):
        super(CNN_Attention_Model, self).__init__()
        self.conv1 = nn.Conv1d(22, 64, kernel_size=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3)
        
        self.pooling1 = nn.MaxPool1d(kernel_size=3)
        self.pooling2 = nn.MaxPool1d(kernel_size=3)
        self.pooling3 = nn.MaxPool1d(kernel_size=3)
        
        self.activation = nn.ReLU()
         
        # was too lazy to implement multi-head from scratch :(
        self.multi_head_attention1 = nn.MultiheadAttention(embed_dim=256, num_heads=16, batch_first=True, dropout=0.5)

        self.dropout = nn.Dropout1d(0.5)
        
        #200*4
        self.fc1 = nn.Linear(1536, out_features=18) 
        self.fc2 = nn.Linear(18, out_features=4)
        
    def forward(self, x):
        # Architecture from paper
        x = self.conv1(x)
        x = self.pooling1(x)
        #x = self.activation(x)
        
        x = self.conv2(x)
        x = self.pooling2(x)
        #x = self.activation(x)
        
        x = self.conv3(x)
        x = self.pooling3(x)
        #x = self.activation(x)
        x = x.transpose(2, 1) #(batch, feature, seq) -> (batch, seq, feature)
        x, _ = self.multi_head_attention1(x, x, x)
        x = self.dropout(x)
        
        #print(x.shape)
        x = x.reshape(-1, 13*256)  # Maintain bsz dimension
        x = self.fc1(x)
        x = F.softmax(self.fc2(x), dim=1)
        return x

    def compute_l1_loss(self):
        l1_loss = 0
        for param in self.parameters():
            l1_loss += torch.abs(param).sum()
        return l1_loss