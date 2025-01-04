#1. Saving state_dict (recommended):
import torch

# Save model state_dict (parameters)
torch.save(model.state_dict(), "model_parameters.pth")



#2. Saving the Entire Model (less recommended):
# Save entire model (architecture + parameters)
torch.save(model, "complete_model.pth")






#1. Loading state_dict:
# Initialize a new model with the same architecture
model = EEGNet(num_input_channels=1, num_classes=5, F1=8, D=2, F2=16)

# Load saved model parameters
model.load_state_dict(torch.load("model_parameters.pth"))
model.eval()  # Set the model to evaluation mode




#2. Loading the Entire Model:
# Load the entire model
model = torch.load("complete_model.pth")
model.eval()  # Set the model to evaluation mode







'''Let’s assume we are comparing two models, one trained with hyperparameters lr=0.001 and F1=8, and the other with lr=0.0005 and F1=16. '''

# First experiment (hyperparameters: lr=0.001, F1=8)
model_1 = EEGNet(num_input_channels=1, num_classes=5, F1=8, D=2, F2=16)
optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=0.001)
train_model(model_1, optimizer_1)  # Your training loop
torch.save(model_1.state_dict(), "model_1_params.pth")

# Second experiment (hyperparameters: lr=0.0005, F1=16)
model_2 = EEGNet(num_input_channels=1, num_classes=5, F1=16, D=2, F2=16)
optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=0.0005)
train_model(model_2, optimizer_2)  # Your training loop
torch.save(model_2.state_dict(), "model_2_params.pth")

# Load both models for comparison
model_1 = EEGNet(num_input_channels=1, num_classes=5, F1=8, D=2, F2=16)
model_1.load_state_dict(torch.load("model_1_params.pth"))

model_2 = EEGNet(num_input_channels=1, num_classes=5, F1=16, D=2, F2=16)
model_2.load_state_dict(torch.load("model_2_params.pth"))

# Evaluate both models on the test dataset
test_accuracy_1 = evaluate_model(model_1, test_loader)
test_accuracy_2 = evaluate_model(model_2, test_loader)

# Compare the results
print(f"Test Accuracy (Model 1): {test_accuracy_1:.4f}")
print(f"Test Accuracy (Model 2): {test_accuracy_2:.4f}")







#>
# Example of saving hyperparameters to a file
import json

hyperparameters = {
    "learning_rate": 0.001,
    "F1": 8,
    "D": 2,
    "F2": 16
}

with open("hyperparameters.json", "w") as f:
    json.dump(hyperparameters, f, indent=4)










