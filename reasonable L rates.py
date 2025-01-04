'''
Steps to Experiment with Different Learning Rates:
Define a Range of Learning Rates:

We'll test at least five learning rates (e.g., 0.001, 0.0001, 0.01, 0.1, 0.5).
Train the Model with Each Learning Rate:

For each learning rate, we'll train the model for a fixed number of epochs (e.g., 10 epochs) and track the training and validation accuracy and loss.
Plot the Results:

We will plot the loss and accuracy for each learning rate to visually determine the best performing one.
Compare Performance:

Based on the results, we can conclude which learning rate yields the best performance for the model.
'''



#1. Train the Model with Different Learning Rates:

import torch.optim as optim
import matplotlib.pyplot as plt

# List of learning rates to test
learning_rates = [0.001, 0.0001, 0.01, 0.1, 0.5]

# Store results for each learning rate
results = {}

# Training function (modified to return accuracy and loss for each learning rate)
def train_with_learning_rate(lr, model, train_loader, valid_loader, num_epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    loss_train_hist, acc_train_hist = [], []
    loss_valid_hist, acc_valid_hist = [], []
    
    for epoch in range(num_epochs):
        model.train()
        loss_train, acc_train = 0, 0
        
        # Training loop
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            
            loss_train += loss.item()
            acc_train += (outputs.argmax(1) == targets).sum().item()
        
        loss_train /= len(train_loader)
        acc_train /= len(train_loader.dataset)
        
        # Validation loop
        model.eval()
        loss_valid, acc_valid = 0, 0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                
                loss_valid += loss.item()
                acc_valid += (outputs.argmax(1) == targets).sum().item()
        
        loss_valid /= len(valid_loader)
        acc_valid /= len(valid_loader.dataset)
        
        # Store history for plotting
        loss_train_hist.append(loss_train)
        acc_train_hist.append(acc_train)
        loss_valid_hist.append(loss_valid)
        acc_valid_hist.append(acc_valid)
    
    return loss_train_hist, acc_train_hist, loss_valid_hist, acc_valid_hist

# Test different learning rates
for lr in learning_rates:
    print(f"Training with learning rate: {lr}")
    model = EEGNetModified()  # You can use EEGNet or EEGNetModified
    loss_train_hist, acc_train_hist, loss_valid_hist, acc_valid_hist = train_with_learning_rate(
        lr, model, train_loader, valid_loader, num_epochs=10)
    
    results[lr] = {
        'loss_train': loss_train_hist,
        'acc_train': acc_train_hist,
        'loss_valid': loss_valid_hist,
        'acc_valid': acc_valid_hist
    }
    
# Plot the results
plt.figure(figsize=(12, 5))

# Plot train and validation loss
plt.subplot(1, 2, 1)
for lr, result in results.items():
    plt.plot(result['loss_train'], label=f'Train LR={lr}')
    plt.plot(result['loss_valid'], label=f'Valid LR={lr}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs Epoch for Different Learning Rates')
plt.legend()

# Plot train and validation accuracy
plt.subplot(1, 2, 2)
for lr, result in results.items():
    plt.plot(result['acc_train'], label=f'Train LR={lr}')
    plt.plot(result['acc_valid'], label=f'Valid LR={lr}')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epoch for Different Learning Rates')
plt.legend()

plt.show()




















