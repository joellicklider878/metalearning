# Meta-Reinforcement Learning Penetration Testing Policy Network

A PyTorch-based Reinforcement Learning (RL) simulation environment and policy network engineered to optimize cybersecurity penetration testing vectors. This project parses historical attack payloads, execution timelines, and defense adaptation responses to train a neural network capable of predicting the most successful exploit actions based on real-world system telemetry.

## Features

- **Custom Penetration Testing Environment (`PenTestEnv`):** Implements an OpenAI Gym-like sequence runner mapped over tabular pentesting event datasets.
- **Dynamic State Normalization:** Automatically processes and rescales physical features—including payload sizing metrics, execution latencies, and threat response scores—to guarantee data distribution uniformity.
- **Deep Policy Network Architecture:** Utilizes a multi-layer feedforward PyTorch policy pipeline leveraging rectified linear activations (`ReLU`) and normalized output probability mappings (`Softmax`).
- **Data Cleanup Integrity Checkers:** Active fallback checks explicitly neutralize missing cells (`NaN`) to prevent catastrophic gradient explosions during network backtracking phases.

## State Space & Action Mappings

### 1. Observation (State) Vectors
The framework processes an environmental observation array containing three rescaled metrics:
- **Payload Size:** Quantized volume distribution (scaled over a baseline parameter of 5,000 bytes).
- **Execution Time:** Machine transaction latency (scaled over a 5.0-second timeout ceiling).
- **Adaptation Score:** Threat defense reaction coefficient.

### 2. Reward Mechanics
The optimization objective forces the model to balance simple objective victories with operational versatility:
$$\text{Reward} = \text{Exploit Success (Binary Flag)} + \text{System Adaptation Score}$$

## Getting Started

### Prerequisites

Ensure you have the required analytical libraries installed in your runtime platform:

```bash
pip install pandas numpy torch
```

### Required Dataset
The runtime agent demands an internal flat source dataset named `meta_reinforcement_pentest_dataset.csv` placed within the execution directory. The file must supply records capturing `Attack type`, `Success`, `Payload size`, `Execution time`, and `Adaption score`.

### Running the Loop

Launch the sequence runner to initialize the environment checks, evaluate the training steps across 100 iterations, and compile your local policy checkpoint:

```bash
python rl-loop.py
```

## Outputs & Artifacts

- **Automated CSV Repairs:** Modifies missing target values directly within `meta_reinforcement_pentest_dataset.csv` to establish strict data completeness.
- **`pentest_policy_model.pt`:** A serialized binary file containing the final weights and structural biases of the trained `SimplePolicyNetwork`. This checkpoint can be dynamically reloaded via `torch.load()` for runtime model scoring or inference.
