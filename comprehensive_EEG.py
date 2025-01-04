'''
EEGNet Architecture Overview
EEGNet typically consists of the following layers:

Conv2D Layer (Layer 1): This is a standard 2D convolution layer with Batch Normalization.
Depthwise Separable Conv2D Layer (Layer 2): This layer performs a depthwise convolution followed by a pointwise convolution.
Average Pooling: Reduces the spatial size.
Dropout Layer: Regularization to prevent overfitting.
Flatten: Converts the output to a 1D tensor.
Fully Connected (Dense) Layer: For classification.
Softmax: For multi-class classification.
'''



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





'''
Conv2D Layer (Layer 1):
This is a basic 2D convolution layer. We use num_input_channels to specify the number of input channels (1 for EEG data as it's usually a 1D signal, but we treat it as a 2D image).
kernel_size_1 specifies the filter size, and kernel_padding_1 ensures that the convolution is applied without changing the input size in the temporal dimension.
Batch Normalization is applied after this layer to stabilize the learning process and speed up convergence.
2. Depthwise Separable Convolution (Layer 2):
Depthwise Separable Convolution consists of two parts:
Depthwise convolution: This is a spatial convolution applied separately for each input channel.
Pointwise convolution: A 1x1 convolution applied to the depthwise outputs.
This layer reduces the number of parameters, making the model more efficient.
We apply Batch Normalization, ELU (Exponential Linear Unit) for activation, and Average Pooling to reduce the spatial size.
Dropout is used after pooling to prevent overfitting.
3. Separable Conv2D (Layer 3):
This layer performs a depthwise convolution followed by a pointwise convolution (as in Layer 2).
The pointwise convolution outputs F2 channels, followed by Batch Normalization and Average Pooling to reduce the size further.
4. Fully Connected Layer (Dense Layer):
After flattening the output from the convolutional layers, the data is passed through a Fully Connected (Dense) Layer,
which reduces the output to the number of num_classes for classification.
5. Softmax:
The Softmax layer is used to convert the output into a probability distribution over the classes.
Hyperparameters:
You can adjust the hyperparameters for F1, F2, D (depth multiplier), kernel_size, and others based on the specific EEG dataset you're working with.

F1 is the number of filters in the first convolutional layer.
D is the depth multiplier applied to F1 for the depthwise separable convolution layers.
F2 is the number of filters in the pointwise convolution in Layer 3.
kernel_size_1, kernel_size_2, etc., control the size of the convolution filters.
dropout_rate is used for regularization to prevent overfitting.
'''







#Example of Using the EEGNet Model:

# Define the model
model = EEGNet(num_input_channels=1, num_classes=5, F1=8, D=2, F2=16)

# Set the model to training mode
model.train()

# Define the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Example input (batch of EEG signals, e.g., 32 samples, 1 channel, 256 time points)
inputs = torch.randn(32, 1, 256, 64)  # Batch of 32, 1 channel, 256 samples, 64 EEG channels

# Forward pass through the model
outputs = model(inputs)

# Compute the loss (using a batch of random labels for demonstration)
labels = torch.randint(0, 5, (32,))  # Random labels for a 5-class problem
loss = loss_fn(outputs, labels)

# Backpropagate the loss and optimize the model
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"Loss: {loss.item()}")









