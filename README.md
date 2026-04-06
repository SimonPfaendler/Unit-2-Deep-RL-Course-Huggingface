# Unit-2-Deep-RL-Course-Huggingface

# Q-Learning Agents: FrozenLake & Taxi

[![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v1.0-green?style=flat&logo=openai&logoColor=white)](https://gymnasium.farama.org/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Deep_RL_Class-yellow)](https://huggingface.co/deep-rl-course/unit2/introduction)

This repository contains my solutions for two classic Reinforcement Learning environments from **Gymnasium**: `FrozenLake-v1` and `Taxi-v3`. 

Both environments were solved using **Q-Learning** (Table-based Reinforcement Learning) as part of the Hugging Face Deep RL Class (Unit 2).

---

## Environment 1: FrozenLake-v1
**Configuration:** 4x4 Grid, No Slippery (`is_slippery=False`)

**Goal:** The agent must navigate a frozen lake from the Start (S) to the Goal (G) without falling into Holes (H).

* **State Space:** 16 (4x4 Grid)
* **Action Space:** 4 (Left, Down, Right, Up)
* **Model on Hugging Face:**
  
  [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/SimonPfaendler/q-FrozenLake-v1-4x4-noSlippery)

![FrozenLake Demo](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit2/frozen_lake.gif)

---

## Environment 2: Taxi-v3
**Goal:** The taxi must pick up a passenger at one of four designated locations and drop them off at the correct destination.

* **State Space:** 500 discrete states.
* **Action Space:** 6 (Move South, North, East, West, Pickup, Dropoff).
* **Model on Hugging Face:**

  [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/SimonPfaendler/Taxi-v3)

![Taxi Demo](https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/unit2/taxi.gif)

---

## How it Works: Q-Learning
Q-Learning is an off-policy value-based algorithm. The agent maintains a **Q-Table** of shape `(state_space, action_space)`, where each cell contains the expected future reward for taking a specific action in a specific state.

The table is updated using the **Bellman Equation**:

$$Q(S, A) \leftarrow Q(S, A) + \alpha [R + \gamma \max_{a} Q(S', a) - Q(S, A)]$$

Where:
* **$Q(S, A)$**: Current Value
* **$\alpha$ (Alpha)**: Learning Rate (how much we accept new value)
* **$R$**: Reward received
* **$\gamma$ (Gamma)**: Discount Factor (importance of future rewards)
* **$\max_{a} Q(S', a)$**: Maximum predicted reward for the next state

---

## Installation & Usage

To run these notebooks locally, you need to install the following dependencies.

```bash
# Basic RL libraries
pip install gymnasium numpy

# Rendering and Saving (Unit 2 Requirements)
# Note: pyyaml fixed to >=6.0.1 for Python 3.12 compatibility
pip install "pyyaml>=6.0.1" imageio imageio_ffmpeg "pyglet==1.5.1" tqdm
