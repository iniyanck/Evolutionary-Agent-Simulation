# Evolutionary Agent Simulation

This project implements a 2D simulation environment where AI agents learn to survive and interact using a Proximal Policy Optimization (PPO) reinforcement learning algorithm. **Inspired by "The Bibites,"** this simulation focuses on evolutionary dynamics where agents exist in a 2D world, interact with food and other agents, and strive for survival. Agents can move, rotate, eat food, and attack other agents, all while trying to maximize their "health points" (HP) and, implicitly, their lifespan.

## Features

* **2D Environment:** A simple 2D plane with configurable width and height, featuring food and agents.
* **Ray-Casting Sight:** Agents perceive their surroundings using ray-casting, detecting food, other agents, and boundaries.
* **Agent Behaviors:**
    * **Movement:** Agents can move forward and rotate.
    * **Eating:** Agents can consume food to regain HP.
    * **Attacking:** Agents can attack other agents, dealing damage.
* **HP System:** Agents have HP that decays over time, increases by eating, and decreases from damage or certain actions.
* **Death and Respawn:** Dead agents drop food and are replaced by new agents, maintaining a consistent population.
* **Proximal Policy Optimization (PPO):** Agents are trained using a PPO algorithm with a Recurrent Neural Network (RNN) policy for handling sequential observations.
* **Pygame Visualization:** A real-time graphical visualization of the simulation, showing agent movement, sight rays, and key statistics.
* **Configurable Simulation Parameters:** Most simulation and training parameters can be adjusted in `config.py`.

## Getting Started

### Prerequisites

* Python 3.8+
* `numpy`
* `torch` (PyTorch)
* `pygame`

You can install the required Python packages using pip:

```bash
pip install numpy torch pygame