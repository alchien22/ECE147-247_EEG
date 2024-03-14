from braindecode.models import EEGConformer
import torch
import torch.nn as nn
from torch.optim import Adam
from utils import train, evaluate
from preprocessing import load_data

train_dataloader, val_dataloader, test_dataloader = load_data(batch_size=32)

model = EEGConformer(n_outputs=4, n_chans=22, final_fc_length=840)
model.to('mps')

optimizer = Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

num_epochs = 100
for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_dataloader, optimizer, criterion, torch.device('mps'))
    val_loss, val_acc = evaluate(model, val_dataloader, criterion, torch.device('mps'))
    print(f"Epoch {epoch + 1}/{num_epochs} loss: {train_loss}, acc: {train_acc}, val_loss: {val_loss}, val_acc: {val_acc}")