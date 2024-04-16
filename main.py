import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import ImageFolder

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # use GPU if available

dataset = ImageFolder('Data', transform=transforms.ToTensor())

# Hyper-parameters
# Subject to change
num_epochs = 5
batch_size = 4
learning_rate = 0.001

train_data, test_data = random_split(dataset, [int(0.8 * len(dataset)), int(0.2 * len(dataset)) + 1])

# TODO: Transform data (preprocessing)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

classes = dataset.classes


# TODO: Implement CNN (class Net(nn.Module))
""" 
    input -> Conv2d -> ReLU -> MaxPool2d -> Conv2d -> ReLU -> MaxPool2d (Feature Learning)
 """
class ConvNet(nn.Module):
    def __init__(self):
        # Convolutional Layers
        self.conv1 = nn.Conv2d(3, 6, 5) # 3 input channels (dependent on colors), 6 output channels, 5x5 kernel size
        self.pool = nn.MaxPool2d(2, 2) # 2x2 kernel size, stride of 2 (shifting pixels by 2)
        self.conv2 = nn.Conv2d(6, 16, 5) # 6 input channels, 16 output channels, 5x5 kernel size (input channel of conv2 must match output of conv1)
        # Fully connected layers
        self.fc1 = nn.Linear(16*5*5, 120) # 16*5*5 is dependent on the input size of the image and what it is scaled to
        self.fc2 = nn.Linear(120, 84) # 120 input features, 84 output features
        self.fc3 = nn.Linear(84, 4) # 84 input features, 4 output features (4 classes) 
    
    def forward(self, x):
        # F.relu is the activation function
        x = self.pool(F.relu(self.conv1(x))) # Flattening
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5) # -1 is a placeholder for the batch size (-1 pytorch will automatically determine the batch size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # no activation function for the output layer
        return x
    

# TODO: Training Loop - remember to push to device so that GPU can be used

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finished Training')

# TODO: Calculate accuracy of model and accuracy of each class
