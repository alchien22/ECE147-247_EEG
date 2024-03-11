import preprocessing
from preprocessing import load_data
from cnn import CNNModel
from cnn_attention import CNN_Attention_Model
from loop import train as training_loop, evaluate
from plot import plot_loss_acc, plot_confusion_matrix
import torch
import torch.nn as nn

train_data_loader, val_data_loader, test_data_loader = load_data(64, True)
model = CNNModel()
criterion = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

history = training_loop(model, train_data_loader, val_data_loader, optim, criterion, "cpu", 10)
avg_loss, acc = evaluate(model, test_data_loader, criterion, "cpu")

#plot_loss_acc(history)
#plot_confusion_matrix(model, val_data_loader, "cpu")

print("Avg Test Loss: ", avg_loss)
print("Test Accuracy: ", acc)