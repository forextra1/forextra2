import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CNN, self).__init__()

        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()

        # Max Pooling Layer 1
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()

        # Max Pooling Layer 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Convolutional Layer 3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        # Max Pooling Layer 3
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully Connected Layer 1
        self.fc1 = nn.Linear(128 * 8 * 8, 1024)  # Example for 32x32 input image size after 3 poolings
        self.relu_fc1 = nn.ReLU()

        # Fully Connected Layer 2
        self.fc2 = nn.Linear(1024, num_classes)

        # Output layer (softmax or direct output layer)
        # Note: CrossEntropyLoss in PyTorch includes softmax internally, so no need to apply it here.
        
    def forward(self, x):
        # Convolutional Layer 1
        x = self.relu1(self.bn1(self.conv1(x)))
        # Max Pooling Layer 1
        x = self.pool1(x)

        # Convolutional Layer 2
        x = self.relu2(self.bn2(self.conv2(x)))
        # Max Pooling Layer 2
        x = self.pool2(x)

        # Convolutional Layer 3
        x = self.relu3(self.bn3(self.conv3(x)))
        # Max Pooling Layer 3
        x = self.pool3(x)

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)  # Flatten

        # Fully Connected Layer 1
        x = self.relu_fc1(self.fc1(x))

        # Fully Connected Layer 2 (output layer)
        x = self.fc2(x)

        return x

# Example Usage:
# Define the model
model = CNN(input_channels=3, num_classes=10)  # Example with RGB input (3 channels) and 10 output classes

# Print model architecture
print(model)


'''
Class Definition:

The class CNN inherits from torch.nn.Module to define a custom neural network. This is the basic structure for defining models in PyTorch.
Convolutional Layers (conv1, conv2, conv3):

nn.Conv2d: 2D convolution layer, where:
in_channels is the number of input channels (e.g., 3 for RGB images).
out_channels is the number of output channels (feature maps) after the convolution.
kernel_size is the size of the convolutional kernel (e.g., 3x3).
stride is how far the kernel moves on the image.
padding is used to add zeros to the image borders to maintain spatial dimensions.
Batch Normalization (bn1, bn2, bn3):
Normalizes the output of the convolutional layers to improve training speed and stability.
ReLU Activation (relu1, relu2, relu3):
The ReLU (Rectified Linear Unit) activation function introduces non-linearity after each convolution operation.
Pooling Layers (pool1, pool2, pool3):

Max Pooling: Reduces spatial dimensions by taking the maximum value over a pool of values (e.g., 2x2).
Helps to reduce computation and the number of parameters by downsampling the spatial size.
Fully Connected Layers (fc1, fc2):

fc1: A dense layer (fully connected) that receives the flattened output from the convolutional layers.
fc2: The output layer that produces the final classification scores. The number of outputs corresponds to the number of classes (e.g., 10 for CIFAR-10).
Flattening:

The output from the last pooling layer is a multi-dimensional tensor. We flatten it into a 1D vector before passing it to the fully connected layers using x.view(x.size(0), -1).
Forward Method:

Defines how data flows through the network during the forward pass:
The input goes through the convolutional layers with activations and pooling.
Then it is flattened and passed through the fully connected layers.
Finally, it produces the output predictions.
Example Usage:
input_channels=3: If the input is an RGB image, it has 3 channels (Red, Green, Blue).
num_classes=10: Example for classification into 10 classes (such as CIFAR-10).
The model is initialized by calling CNN(input_channels=3, num_classes=10).
'''






'''
Next Steps:
Optimizer: You can use optimizers like Adam, SGD, etc., for training the network.
Loss Function: For multi-class classification, you can use CrossEntropyLoss.
Training Loop: You'll need to write a loop to train and test the model.
'''

import torch.optim as optim

# Define model, optimizer, and loss function
model = CNN(input_channels=3, num_classes=10)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Example training loop
for epoch in range(10):  # Number of epochs
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/10], Loss: {running_loss/len(train_loader)}')









'''

To use the CNN model that we defined, you'll need to follow these steps:

Load Your Dataset
Preprocess the Data
Define the Model, Loss Function, and Optimizer
Train the Model
Evaluate the Model
Use the Model for Inference/Prediction

'''

#Load Your Dataset
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define transformation for data augmentation and normalization
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create DataLoader for train and test datasets
train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
test_loader = DataLoader(testset, batch_size=32, shuffle=False)





#Preprocess the Data



#Define the Model, Loss Function, and Optimizer
import torch.optim as optim
import torch.nn as nn

# Define the model (CIFAR-10 images have 3 channels (RGB) and 10 classes)
model = CNN(input_channels=3, num_classes=10)

# Define the loss function (cross-entropy loss for classification)
loss_fn = nn.CrossEntropyLoss()

# Define the optimizer (Adam optimizer in this case)
optimizer = optim.Adam(model.parameters(), lr=0.001)

'''
Loss Function: CrossEntropyLoss is commonly used for multi-class classification tasks.
Optimizer: Adam is used here, but you could also use SGD or other optimizers.

'''







#Train the Model
num_epochs = 10  # Number of epochs

# Training Loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()  # Zero the gradients

        outputs = model(inputs)  # Forward pass
        loss = loss_fn(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagation

        optimizer.step()  # Update the weights

        running_loss += loss.item()  # Accumulate loss

    # Print statistics for every epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
'''
model.train() sets the model to training mode, which is required for certain layers like Dropout and BatchNorm.
optimizer.zero_grad() clears previous gradients.
outputs = model(inputs) performs a forward pass through the network.
loss.backward() computes the gradients.
optimizer.step() updates the model's parameters.
'''







#Evaluate the Model
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0

# Evaluate on the test data
with torch.no_grad():  # Disable gradient calculation
    for inputs, labels in test_loader:
        outputs = model(inputs)  # Forward pass
        _, predicted = torch.max(outputs, 1)  # Get the predicted class
        total += labels.size(0)  # Number of samples
        correct += (predicted == labels).sum().item()  # Correct predictions

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')

'''
model.eval() sets the model to evaluation mode, which is important for layers like Dropout and BatchNorm.
torch.no_grad() ensures that gradients are not calculated during evaluation, saving memory and computation.
We calculate the accuracy by comparing the predicted labels with the true labels.
'''







#Use the Model for Inference/Prediction
# Load a single image and apply the same transformation
image = transform(some_image)  # Replace `some_image` with your input image (as a PIL image)
image = image.unsqueeze(0)  # Add a batch dimension (1, C, H, W)

# Inference
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    output = model(image)
    _, predicted_class = torch.max(output, 1)

print(f'Predicted Class: {predicted_class.item()}')

'''
transform(some_image): Preprocess the input image (make sure it's a PIL image).
image.unsqueeze(0): Add a batch dimension as the model expects batches of images.
model(image): Perform a forward pass to get the class scores.
torch.max(output, 1): Extract the predicted class by finding the class with the highest score.
'''



#
'''
Model Saving/Loading: After training, you can save your model weights with torch.save(model.state_dict(), 'model.pth'),
and load it back later with model.load_state_dict(torch.load('model.pth')).

'''








#Accuracy
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')





#Precision, Recall, F1-Score
from torchmetrics.classification import Precision, Recall, F1Score

# Initialize metrics
precision = Precision(task='multiclass', num_classes=num_class)
recall = Recall(task='multiclass', num_classes=num_class)
f1_score = F1Score(task='multiclass', num_classes=num_class)

# Evaluate on the test data
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        precision(outputs, labels)
        recall(outputs, labels)
        f1_score(outputs, labels)

# Compute and print results
print(f'Precision: {precision.compute().item():.4f}')
print(f'Recall: {recall.compute().item():.4f}')
print(f'F1-Score: {f1_score.compute().item():.4f}')




#Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())  # Store predictions
        all_labels.extend(labels.cpu().numpy())  # Store true labels

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=trainset.classes, yticklabels=trainset.classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()






#Log-Loss (Cross-Entropy Loss)
import torch.nn.functional as F

log_loss = F.cross_entropy(outputs, labels)





#ROC Curve and AUC (Area Under the Curve)
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# Binarize the labels for multi-class classification
y_true = label_binarize(all_labels, classes=range(num_class))
y_score = model(torch.tensor(inputs).float()).cpu().detach().numpy()

# Compute ROC curve and AUC for each class
fpr, tpr, _ = roc_curve(y_true.ravel(), y_score.ravel())
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()







#Putting It All Together

from torchmetrics import Accuracy, Precision, Recall, F1Score
from sklearn.metrics import confusion_matrix

# Initialize metrics
accuracy_metric = Accuracy(task='multiclass', num_classes=num_class)
precision_metric = Precision(task='multiclass', num_classes=num_class)
recall_metric = Recall(task='multiclass', num_classes=num_class)
f1_metric = F1Score(task='multiclass', num_classes=num_class)

all_preds = []
all_labels = []

# Evaluate the model
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)

        # Update metrics
        accuracy_metric(outputs, labels)
        precision_metric(outputs, labels)
        recall_metric(outputs, labels)
        f1_metric(outputs, labels)

        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Print metrics
print(f'Accuracy: {accuracy_metric.compute().item():.4f}')
print(f'Precision: {precision_metric.compute().item():.4f}')
print(f'Recall: {recall_metric.compute().item():.4f}')
print(f'F1-Score: {f1_metric.compute().item():.4f}')

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=trainset.classes, yticklabels=trainset.classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()













