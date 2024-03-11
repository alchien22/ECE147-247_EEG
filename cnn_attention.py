import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, x):
        # get the query, key and value vectors
        x = x.transpose(2, 1)
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        
        # calculate the attention scores by the matrix mult. of queries and keys
        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)

        # convert scores into probabilities w/ softmax
        attention = self.softmax(scores)
        
        # get the weighted values by the matrix mult. of attention scores and values
        weighted = torch.bmm(attention, values)
        
        return weighted

class CNN_Attention_Model(nn.Module):
    def __init__(self):
        super(CNN_Attention_Model, self).__init__()
        self.conv1 = nn.Conv1d(22, 64, kernel_size=3, padding=1)
        self.self_attention1 = SelfAttention(input_dim=25)
        # was too lazy to implement multi-head from scratch :(
        self.multi_head_attention1 = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.batch_norm1 = nn.BatchNorm1d(num_features=400)
        self.activation1 = nn.ReLU()
        self.pooling1 = nn.MaxPool1d(kernel_size=3, padding=1)
        
        #200*4
        self.fc = nn.Linear(400*22, out_features=4) 
        
    def forward(self, x):
        # Architecture from paper
        x = self.conv1(x)
        #print(x.shape)
        x = x.transpose(2, 1) #(batch, feature, seq) -> (batch, seq, feature)
        x, _ = self.multi_head_attention1(x, x, x)
        x = self.batch_norm1(x)
        x = self.activation1(x)
        x = self.pooling1(x)
        #print("After pooling:", x.shape)
        x = x.view(-1, 400*22)  # Maintain bsz dimension
        #print("hello1")
        x = F.softmax(self.fc(x), dim=1)
        #print("hello2")
        return x
