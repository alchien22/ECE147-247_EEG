import torch
import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

#from torch.utils.tensorboard import SummaryWriter

def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs):
    """
    Train the MLP classifier on the training set and evaluate it on the validation set every epoch.

    Args:
        model (MLP): MLP classifier to train.
        train_loader (torch.utils.data.DataLoader): Data loader for the training set.
        val_loader (torch.utils.data.DataLoader): Data loader for the validation set.
        optimizer (torch.optim.Optimizer): Optimizer to use for training.
        criterion (callable): Loss function to use for training.
        device (torch.device): Device to use for training.
        num_epochs (int): Number of epochs to train the model.
    """
    # Place model on device
    model = model.to(device)
    
    #train_writer = SummaryWriter('logs/train')
    #val_writer = SummaryWriter('logs/validation')
    
    history = {'train_loss': [], 'train_accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
    
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Use tqdm to display a progress bar during training
        with tqdm.tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}') as pbar:
            for inputs, labels in train_loader:
                #get rid of the one-hot encoding, and just get the class indices of the label
                #print(labels)
                labels = torch.argmax(labels, dim=1)
                
                # Move inputs and labels to device
                inputs = inputs.to(device)
                labels = labels.to(device)
               
                # Zero out gradients
                optimizer.zero_grad()
              
                # Compute the logits and loss
                model = model.double()
                logits = model(inputs)
               
                loss = criterion(logits.float(), labels.long())

                # Backpropagate the loss
                loss.backward()

                # Update the weights
                optimizer.step()

                train_loss += loss.item()
                # Update the progress bar
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
                
                _, predictions = torch.max(logits, dim=1)
                train_correct += (predictions == labels).sum().item()
                train_total += labels.size(0)
                
                pbar.set_postfix(loss=train_loss / train_total, accuracy=train_correct / train_total)

        # Evaluate the model on the validation set
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        print(f'Validation set: Average loss = {val_loss:.4f}, Accuracy = {val_accuracy:.4f}')
        
        # Log the validation loss and accuracy for TensorBoard visualization
        #val_writer.add_scalar('Loss', avg_loss, epoch)
        #val_writer.add_scalar('Accuracy', accuracy, epoch)
        
        # Update history
        history['train_loss'].append(train_loss / train_total)
        history['train_accuracy'].append(train_correct / train_total)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        
    # Close the SummaryWriter objects
    #train_writer.close()
    #val_writer.close()
    return history

def evaluate(model, test_loader, criterion, device):
    """
    Evaluate the MLP classifier on the test set.

    Args:
        model (MLP): MLP classifier to evaluate.
        test_loader (torch.utils.data.DataLoader): Data loader for the test set.
        criterion (callable): Loss function to use for evaluation.
        device (torch.device): Device to use for evaluation.

    Returns:
        float: Average loss on the test set.
        float: Accuracy on the test set.
    """
    model.eval()  # Set model to evaluation mode
    #test_writer = SummaryWriter('logs/test')
    with torch.no_grad():
        total_loss = 0.0
        num_correct = 0
        num_samples = 0

        for inputs, labels in test_loader:
            # Move inputs and labels to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = torch.argmax(labels, dim=1)
            model = model.double()
            
            # Compute the logits and loss
            logits = model(inputs)
            loss = criterion(logits, labels.long())
            total_loss += loss.item()

            # Compute the accuracy
            _, predictions = torch.max(logits, dim=1)
            #print("predictions: ", predictions)
            #print("labels", labels)
            num_correct += (predictions == labels).sum().item()
            num_samples += len(inputs)

    # Compute the average loss and accuracy
    avg_loss = total_loss / len(test_loader)
    accuracy = num_correct / num_samples
    
    # Log the testing loss and accuracy for TensorBoard visualization
    #test_writer.add_scalar('Loss', avg_loss, epoch)
    #test_writer.add_scalar('Accuracy', accuracy, epoch)


    return avg_loss, accuracy

def plot_confusion_matrix(model, test_loader, device):
    model.eval()  # Set model to evaluation mode
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move inputs and labels to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = torch.argmax(labels, dim=1)
            model = model.double()

            # Compute the logits and predictions
            logits = model(inputs)
            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(cm))
    plt.xticks(tick_marks, [str(i) for i in range(len(cm))], rotation=45)
    plt.yticks(tick_marks, [str(i) for i in range(len(cm))])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.show()