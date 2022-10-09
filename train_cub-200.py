import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from torch import nn
import torchvision
from train_val_functions import train_one_epoch, validation

from dataset_and_data_loader import train_loader, test_loader

# Train model > Select best Hyper-parameters from 'hyper_parameters_tuning.py'

# Model: EfficientNet
model = torchvision.models.efficientnet_b0()
model.classifier[1] = nn.Linear(1280, 200)

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

lr = 0.1
wd = 1e-4
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)

# History list
loss_train_hist = []
loss_valid_hist = []

acc_train_hist = []
acc_valid_hist = []

best_loss_valid = torch.inf
epoch_counter = 0

num_epochs = 5
if __name__ == "__main__":
    for epoch in range(num_epochs):
        # Train
        model, loss_train, acc_train = train_one_epoch(model,
                                                       train_loader,
                                                       loss_fn,
                                                       optimizer,
                                                       epoch)
        # Validation
        loss_valid, acc_valid = validation(model,
                                           test_loader,
                                           loss_fn)

        loss_train_hist.append(loss_train)
        loss_valid_hist.append(loss_valid)

        acc_train_hist.append(acc_train)
        acc_valid_hist.append(acc_valid)

        if loss_valid < best_loss_valid:
            torch.save(model, 'model.pt')
            best_loss_valid = loss_valid

        print(f'Valid: Loss = {loss_valid:.4}, Acc = {acc_valid:.4}')
        print()

        epoch_counter += 1

# Plot Loss
plt.plot(range(epoch_counter), loss_train_hist, 'r-', label='Train')
plt.plot(range(epoch_counter), loss_valid_hist, 'b-', label='Validation')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.grid(True)
plt.legend()
plt.savefig('loss')

# Plot Accuracy
plt.plot(range(epoch_counter), acc_train_hist, 'r-', label='Train')
plt.plot(range(epoch_counter), acc_valid_hist, 'b-', label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.grid(True)
plt.legend()
plt.savefig('acc')
