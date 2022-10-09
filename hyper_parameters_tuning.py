from torch.utils.data import random_split, DataLoader
import torch.optim as optim
from torch import nn
import torch
import torchvision

from dataset_and_data_loader import train_set
from train_val_functions import train_one_epoch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = torchvision.models.efficientnet_b0()
model.classifier[1] = nn.Linear(1280, 200)
model = model.to(device)
# model(torch.randn(10, 3, 224, 224)).shape


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000

print(count_parameters(model))



# Loss & Optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# Efficient way for set hyperparams

# Step 1: check forward path
# Calculate loss for one batch

x_batch, y_batch = next(iter(train_loader))
outputs = model(x_batch.to(device))
loss = loss_fn(outputs, y_batch.to(device))
# print(loss)

# Step 2: check backward path
# 
# Select some random batches and train the model

_, mini_train_dataset = random_split(train_set, (len(train_set) - 1000, 1000))
mini_train_loader = DataLoader(mini_train_dataset, 10)

num_epochs = 20
for epoch in range(num_epochs):
    model, _, _ = train_one_epoch(model, mini_train_loader, loss_fn, optimizer, epoch)

# Step 3: select best lr
# 
# Train all data for one epoch


num_epochs = 2
for lr in [0.1, 0.01, 0.001]:
    print(f'LR={lr}')
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4, momentum=0.9)
    for epoch in range(num_epochs):
        model, _, _ = train_one_epoch(model, train_loader, loss_fn, optimizer, epoch)
    print()

# Step 4: Tuning LR and weight_decay(WD)
#
# Create a small grid based on the WD and the best LR

num_epochs = 5

for lr in [0.08, 0.09, 0.1, 0.15, 0.2]:
    for wd in [1e-4, 1e-5, 0.]:
        model = model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        print(f'LR={lr}, WD={wd}')

        for epoch in range(num_epochs):
            model, loss, _ = train_one_epoch(model, train_loader, loss_fn, optimizer, epoch)
        print()
