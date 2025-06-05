import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Load dataset
df = pd.read_csv("meta_reinforcement_pentest_dataset.csv")

# Ensure required columns exist, and fill missing values with 0
required_cols = ["Execution time", "Adaption score"]
for col in required_cols:
    if col not in df.columns:
        df[col] = 0  # Initialize with default value

# Save updated dataset
df.to_csv("meta_reinforcement_pentest_dataset.csv", index=False)

# Custom pentest environment
class PenTestEnv:
    def __init__(self):
        self.action_space_size = len(df["Attack type"].unique())
        self.observation_space_shape = (3,)
        self.state = None
        self.current_step = 0

    def reset(self):
        """Reset environment and return initial observation."""
        self.current_step = np.random.randint(0, len(df))
        state_values = df.loc[self.current_step, ["Payload size", "Execution time", "Adaption score"]].fillna(0).values / np.array([5000, 5.0, 1.0])
        self.state = torch.tensor(state_values, dtype=torch.float32).numpy()

        # Prevent NaN states
        if np.isnan(self.state).any():
            self.state = np.zeros_like(self.state, dtype=np.float32)

        return self.state

    def step(self, action):
        attack_type = df.iloc[self.current_step]["Attack type"]
        success = df.iloc[self.current_step]["Success"]

        reward = success + df.iloc[self.current_step]["Adaption score"]
        done = self.current_step >= len(df) - 1
        self.current_step += 1

        if not done:
            state_values = df.loc[self.current_step, ["Payload size", "Execution time", "Adaption score"]].fillna(0).values / np.array([5000, 5.0, 1.0])
            self.state = torch.tensor(state_values, dtype=torch.float32).numpy()

        # Prevent NaN states
        if np.isnan(self.state).any():
            self.state = np.zeros_like(self.state, dtype=np.float32)

        return self.state, reward, done

# Define model
class SimplePolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimplePolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)

# Training setup
env = PenTestEnv()
obs_dim = env.observation_space_shape[0]
action_dim = env.action_space_size

policy = SimplePolicyNetwork(obs_dim, action_dim)
optimizer = optim.Adam(policy.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_probs = policy(state_tensor)
        action = torch.argmax(action_probs).item()

        next_state, reward, done = env.step(action)
        total_reward += reward

        # Compute loss (dummy loss for basic training)
        target = torch.tensor([action], dtype=torch.long)
        loss = loss_fn(action_probs.unsqueeze(0), target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}: Total Reward = {total_reward}")

# Save trained model
torch.save(policy.state_dict(), "pentest_policy_model.pt")
print("Training complete! Model saved as 'pentest_policy_model.pt'.")
