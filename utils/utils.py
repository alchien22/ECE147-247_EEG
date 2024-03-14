import torch
from tqdm import tqdm

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for inputs, labels in tqdm(dataloader):
        inputs = inputs.float().to(device)
        labels = torch.argmax(labels, dim=1)
        labels = labels.long().to(device)
        
        optimizer.zero_grad()

        logits = model(inputs)

        loss = criterion(logits, labels)
        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        
        preds = torch.argmax(logits, dim=1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    return train_loss / len(dataloader), train_correct / train_total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    for inputs, labels in tqdm(dataloader):
        inputs = inputs.float().to(device)
        labels = torch.argmax(labels, dim=1)
        labels = labels.long().to(device)
        
        with torch.no_grad():
            logits = model(inputs)
            loss = criterion(logits, labels)

        val_loss += loss.item()
        
        preds = torch.argmax(logits, dim=1)
        val_correct += (preds == labels).sum().item()
        val_total += labels.size(0)

    return val_loss / len(dataloader), val_correct / val_total
