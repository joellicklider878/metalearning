import numpy as np
import torch

# Load pre-trained PyTorch model
attacker_model = torch.load("saved_model.pt")

def mutate_attack_strategy(current_strategy):
    mutation_probability = 0.1
    attack_types = ["SQL Injection", "Privilege Escalation", "Phishing", "Zero-Day exploit"] 
    if np.random.rand() < mutation_probability:
        new_strategy = np.random.choice(attack_types)
        print(f"Strategy mutated from {current_strategy} to {new_strategy}")
        return new_strategy
    return current_strategy

def calculate_reward(success, stealth_score, detection_time, adaptation_factor):
    max_detection_time = 10
    normalized_detection_time = min(detection_time, max_detection_time) / max_detection_time
    attack_reward = (success * stealth_score) - adaptation_factor  # Reward stealth, penalize predictability
    defense_reward = (1 - success) * (1 / (detection_time + 1))  # Detecting faster yields higher reward
    return attack_reward, defense_reward

def train_adversarial_agents(attacker_model, defender_model, env, timesteps=5000):
    for _ in range(timesteps):
        attacker_obs = env.reset()
        defender_obs = env.reset()
        
        attack_action = attacker_model.predict(attacker_obs, deterministic=True)[0]
        defense_action = defender_model.predict(defender_obs, deterministic=True)[0]
        
        attacker_obs, attack_reward, done, _ = env.step(attack_action)
        defender_obs, defense_reward, done, _ = env.step(defense_action)

        attacker_model.learn(total_timesteps=100)
        defender_model.learn(total_timesteps=100)

    return attacker_model, defender_model

def continual_training(attacker_model, defender_model, env, retrain_interval=50):
    step = 0
    while step < 50:
        if step % retrain_interval == 0:
            attacker_model.learn(total_timesteps=200)
            defender_model.learn(total_timesteps=200)
            print(f"Retrained models at step {step}")
        step += 1
