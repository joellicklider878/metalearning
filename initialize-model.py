import torch
import torch.nn as nn

# Correct model architecture to match the saved state_dict
class PentestPolicyModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim1=128, hidden_dim2=128, output_dim=4):
        super(PentestPolicyModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)  # Must match (128, 3)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)  # Must match (128, 128)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)  # Must match (4, 128)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model with matching architecture
attacker_model = PentestPolicyModel()

# Load the saved state dictionary
attacker_model.load_state_dict(torch.load("pentest_policy_model.pt"))

# Set model to evaluation mode
attacker_model.eval()

print("Model loaded successfully!")
