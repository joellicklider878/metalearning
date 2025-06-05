import pandas as pd
import numpy as np
import torch
import pickle
import os
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)
#define the path
model_path = r"d:\py-projects\metalearning\saved_model.pt"

# Load pre-trained models
with open("attacker.pkl", "rb") as f:
    model = pickle.load(f)

with open("defender.pkl", "rb") as f:
    model = pickle.load(f)

# define MyModel class

class MyModel(torch.nn.Module):
    def __init__(self, attack_data=None, defend_data=None):
        super(MyModel, self).__init__()
        self.attack = attack_data if attack_data is not None else {}
        self.defend = defend_data if defend_data is not None else {}

    def forward(self, x):
        # Define the forward pass if needed
        return x

    def save_model(self, filename="model.pt"):
        torch.save(self, filename)
        print(f"Model saved as {filename}")

# Example usage
model = MyModel(attack_data={"power": 80, "speed": 90}, defend_data={"shield": 70, "resistance": 85})
model.save_model()

#extract parameters

# Load the model
with torch.serialization.safe_globals([MyModel]):
    model = torch.load("model.pt", weights_only=True)
model.eval()  # Set to evaluation mode if needed

# Extract state_dict parameters
model.load_state_dict(torch.load("model.pt"), strict=False)

# Define the parameters to extract
target_params = ["fc1.weight", "fc2.bias", "fc2.weight", "fc3.bias", "fc3.weight", "fc3.bias"]

# Print extracted values
for param in target_params:
    if param in state_dict:
        print(f"Parameter: {param}\nValues:\n{state_dict[param]}\n")
    else:
        print(f"Warning: {param} not found in model's state_dict.")

#convert
# Load the model
model = torch.load("model.pt")
model.eval()  # Set to evaluation mode if needed

# Extract state_dict parameters and convert to NumPy arrays
state_dict = model.state_dict()
numpy_params = {name: param.cpu().numpy() for name, param in state_dict.items()}

# Print converted parameters
for name, values in numpy_params.items():
    print(f"Parameter: {name}\nValues:\n{values}\n")


# Return attack and defense predictions as a dictionary
   
# Load the model
model = torch.load("model.pt")
model.eval()  # Set to evaluation mode if needed

# Example input (Modify according to your model's expected input shape)
example_input = torch.rand(1, model.input_size)  # Assuming `input_size` exists

# Perform predictions
output = model(example_input)

# Extract attack and defense values (Modify based on your model's output structure)
attack_prediction = output[0].item()  # Assuming first output represents attack
defense_prediction = output[1].item()  # Assuming second output represents defense

# Return predictions as a dictionary
predictions = {
    "attack": attack_prediction,
    "defense": defense_prediction
}

print(predictions) 

# Generate adversarial scenarios dynamically

def generate_adversarial_example(model, input_data, target_label, epsilon=0.01):
    """
    Generates an adversarial example using the FGSM attack.
    
    :param model: Trained PyTorch model.
    :param input_data: Original input tensor.
    :param target_label: True label of the input.
    :param epsilon: Perturbation magnitude.
    :return: Adversarial example tensor.
    """
    # Set model to evaluation mode
    model.eval()

    # Ensure input requires gradient for attack
    input_data.requires_grad = True

    # Forward pass
    output = model(input_data)
    loss = torch.nn.functional.cross_entropy(output, target_label)

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Generate adversarial perturbation
    adversarial_example = input_data + epsilon * input_data.grad.sign()
    adversarial_example = torch.clamp(adversarial_example, 0, 1)  # Ensure valid range

    return adversarial_example

# Example usage
# Assuming 'model' is a trained PyTorch model and 'input_tensor' is a valid input
# target_label should be a tensor containing the correct label index
# adversarial_sample = generate_adversarial_example(model, input_tensor, target_label)

# Generate scenario dataset

# Define the number of scenarios
num_scenarios = 100

# Generate random attack and defense values
np.random.seed(42)  # For reproducibility
attack_values = np.random.randint(50, 100, size=num_scenarios)
defense_values = np.random.randint(50, 100, size=num_scenarios)

# Create scenario dataset
scenario_data = pd.DataFrame({
    "Scenario_ID": range(1, num_scenarios + 1),
    "Attack": attack_values,
    "Defense": defense_values
})

# Save to CSV
scenario_data.to_csv("scenario_dataset.csv", index=False)

print("Scenario dataset created and saved as scenario_dataset.csv!")
