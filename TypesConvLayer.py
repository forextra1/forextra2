'''
Modify the EEGNet model:

Replace the Depthwise Conv2D and Separable Conv2D layers with standard Conv2d layers while keeping the kernel sizes, number of filters, and padding the same.
Train and evaluate the new model:

We'll then train both the original EEGNet (with Depthwise and Separable convolutions) and the modified EEGNet (with standard convolutions) on the same dataset.
Track the total number of parameters in both models.
Compare their performance (accuracy, loss, etc.).
Save the results:

Save the notebook as Experiment 2.ipynb.
'''





#1. Modified EEGNet Model (with Standard Convolutions): >>>>>>>>>>>>>>>>>>>>>>>>>>>
#We will replace the Depthwise Conv2D and Separable Conv2D layers with standard Conv2d layers.

import torch
import torch.nn as nn

class EEGNetModified(nn.Module): 
    def __init__(self, num_input_channels=1, num_classes=5, F1=8, D=2, F2=16, kernel_size_1=64, kernel_size_2=16, kernel_size_3=8, kernel_size_4=8):
        super(EEGNetModified, self).__init__()
        
        # Layer 1 (Standard Convolution instead of Depthwise)
        self.conv2d = nn.Conv2d(num_input_channels, F1, kernel_size=kernel_size_1, padding=kernel_size_1 // 2)
        self.batch_norm_1 = nn.BatchNorm2d(F1)
        
        # Layer 2 (Standard Convolution instead of Depthwise and Separable)
        self.conv2d_2 = nn.Conv2d(F1, D * F1, kernel_size=kernel_size_2, padding=kernel_size_2 // 2)
        self.batch_norm_2 = nn.BatchNorm2d(D * F1)
        self.elu = nn.ELU()
        self.avg_pool_1 = nn.AvgPool2d(kernel_size=2)
        self.dropout = nn.Dropout2d(0.25)
        
        # Layer 3 (Standard Convolution instead of Separable Convolutions)
        self.conv2d_3 = nn.Conv2d(D * F1, D * F1, kernel_size=kernel_size_3, padding=kernel_size_3 // 2)
        self.conv2d_4 = nn.Conv2d(D * F1, F2, kernel_size=kernel_size_4)
        self.batch_norm_3 = nn.BatchNorm2d(F2)
        self.avg_pool_2 = nn.AvgPool2d(kernel_size=2)
        
        # Layer 4 (Fully connected layer after Flatten)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(F2 * 4, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Layer 1
        x = self.batch_norm_1(self.conv2d(x))
        
        # Layer 2
        x = self.batch_norm_2(self.conv2d_2(x))
        x = self.elu(x)
        x = self.dropout(self.avg_pool_1(x))
        
        # Layer 3
        x = self.conv2d_3(x)
        x = self.batch_norm_3(self.conv2d_4(x))
        x = self.elu(x)
        x = self.dropout(self.avg_pool_2(x))
        
        # Layer 4
        x = self.flatten(x)
        x = self.dense(x)
        x = self.softmax(x)
        
        return x








#Use the previous EEGNet model with Depthwise and Separable layers.
import torch
import torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self, num_input_channels=1, num_classes=5, F1=8, D=2, F2=16, 
                 kernel_size_1=(1, 64), kernel_size_2=(F1, 8), kernel_size_3=(F1, 16), kernel_size_4=(F1, 1),
                 kernel_padding_1=(0, 32), kernel_padding_2=(0, 4), kernel_padding_3=(0, 8), 
                 kernel_avgpool_1=(1, 4), kernel_avgpool_2=(1, 8), dropout_rate=0.25, signal_length=256):
        super(EEGNet, self).__init__()
        
        # First layer - Conv2D + BatchNorm
        self.conv2d = nn.Conv2d(num_input_channels, F1, kernel_size_1, padding=kernel_padding_1)
        self.Batch_norm_1 = nn.BatchNorm2d(F1)

        # Second layer - Depthwise Separable Conv2D + BatchNorm + ELU + AvgPool + Dropout
        self.Depthwise_conv2d = nn.Conv2d(F1, D * F1, kernel_size_2, groups=F1)
        self.Batch_norm_2 = nn.BatchNorm2d(D * F1)
        self.Elu = nn.ELU()
        self.AvgPool_1 = nn.AvgPool2d(kernel_avgpool_1)
        self.Dropout_1 = nn.Dropout2d(dropout_rate)

        # Third layer - Separable Conv2D + BatchNorm + AvgPool
        self.Separable_conv2d_depth = nn.Conv2d(D * F1, D * F1, kernel_size_3, padding=kernel_padding_3, groups=D * F1)
        self.Separable_conv2d_point = nn.Conv2d(D * F1, F2, kernel_size_4)
        self.Batch_norm_3 = nn.BatchNorm2d(F2)
        self.AvgPool_2 = nn.AvgPool2d(kernel_avgpool_2)

        # Fully connected layer and output
        self.Flatten = nn.Flatten()
        self.Dense = nn.Linear(F2 * (signal_length // 32), num_classes)
        self.Softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Layer 1 - Convolution and Batch Normalization
        x = self.Batch_norm_1(self.conv2d(x))

        # Layer 2 - Depthwise Separable Convolution, Batch Normalization, ELU, Avg Pooling, and Dropout
        x = self.Batch_norm_2(self.Depthwise_conv2d(x))
        x = self.Elu(x)
        x = self.Dropout_1(self.AvgPool_1(x))

        # Layer 3 - Separable Convolution, Batch Normalization, and Avg Pooling
        x = self.Separable_conv2d_depth(x)
        x = self.Batch_norm_3(self.Separable_conv2d_point(x))
        x = self.Elu(x)
        x = self.AvgPool_2(x)

        # Flattening and passing through Dense layer
        x = self.Flatten(x)
        x = self.Dense(x)
        x = self.Softmax(x)

        return x









#3.Counting the Number of Parameters in the Models:
from torchsummary import summary

# Initialize the original EEGNet model
original_model = EEGNet(num_input_channels=1, num_classes=5, F1=8, D=2, F2=16)

# Initialize the modified EEGNet model
modified_model = EEGNetModified(num_input_channels=1, num_classes=5, F1=8, D=2, F2=16)

# Print the summary of both models to see the number of parameters
print("Original EEGNet Model Summary:")
summary(original_model, (1, 64, 64))  # assuming the input is of shape (1, 64, 64)

print("\nModified EEGNet Model Summary:")
summary(modified_model, (1, 64, 64))  # assuming the input is of shape (1, 64, 64)








#4. Training Both Models:
import torch.optim as optim

# Initialize the models
original_model = EEGNet(num_input_channels=1, num_classes=5, F1=8, D=2, F2=16)
modified_model = EEGNetModified(num_input_channels=1, num_classes=5, F1=8, D=2, F2=16)

# Define the optimizer and loss function
optimizer = optim.Adam(original_model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Train both models using the same dataset
def train_and_evaluate(model, train_loader, valid_loader, test_loader):
    model.train()
    for epoch in range(10):  # Train for 10 epochs
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
        
    # Evaluate on the test set
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = correct / total
    return accuracy

# Assuming train_loader, valid_loader, test_loader are defined
accuracy_original = train_and_evaluate(original_model, train_loader, valid_loader, test_loader)
accuracy_modified = train_and_evaluate(modified_model, train_loader, valid_loader, test_loader)

print(f"Original EEGNet Accuracy: {accuracy_original * 100:.2f}%")
print(f"Modified EEGNet Accuracy: {accuracy_modified * 100:.2f}%")













