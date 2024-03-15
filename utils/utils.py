import torch
from tqdm import tqdm
from utils.preprocessing import load_data
from utils.plot import plot_loss_acc
from utils.seed import seed_everything

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

def test(model, dataloader, device):
    model.eval()
    
    test_correct = 0
    test_total = 0

    for inputs, labels in tqdm(dataloader):
        inputs = inputs.float().to(device)
        labels = torch.argmax(labels, dim=1)
        labels = labels.long().to(device)
        
        with torch.no_grad():
            logits = model(inputs)
        
        preds = torch.argmax(logits, dim=1)
        test_correct += (preds == labels).sum().item()
        test_total += labels.size(0)

    return test_correct / test_total

def fit(model, optimizer, criterion, num_epochs, device):
    seed_everything(0)

    train_dataloader, val_dataloader, _ = load_data(batch_size=32)

    history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_dataloader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_dataloader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        print(f"Epoch {epoch + 1}/{num_epochs} loss: {train_loss}, acc: {train_acc}, val_loss: {val_loss}, val_acc: {val_acc}")
        if (epoch + 1) % 5 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            }, f'gru_epoch{epoch + 1}.pt')
        
    plot_loss_acc(history)