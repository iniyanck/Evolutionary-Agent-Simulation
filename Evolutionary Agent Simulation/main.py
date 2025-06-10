# main.py

import torch
import numpy as np
import collections
import pygame
import threading
import queue
from torch.distributions import Bernoulli, Normal

from environment import Environment
from agent import Agent
from network import AgentRNN
from training import PPO
from visualization import SimulationRenderer
from config import SimulationConfig

def training_worker(ppo_agent, training_queue):
    while True:
        experiences_batch = training_queue.get()
        if experiences_batch is None:
            print("Training worker received stop signal. Exiting.")
            break
        
        if experiences_batch:
            try:
                ppo_agent.train(experiences_batch)
            except Exception as e:
                print(f"Error during PPO training: {e}")
        training_queue.task_done()

def run_simulation():
    env = Environment()
    renderer = SimulationRenderer(SimulationConfig.PLANE_WIDTH, SimulationConfig.PLANE_HEIGHT)

    input_dim = SimulationConfig.NUM_SIGHT_RAYS * 3 + 1
    output_dim = 7 

    ppo_agent = PPO(input_dim, output_dim)

    all_experiences = collections.defaultdict(lambda: collections.defaultdict(list))
    agent_hidden_states = {agent_id: ppo_agent.policy_net.init_hidden() for agent_id in env.agents.keys()}
    agent_lifespans = collections.defaultdict(int)
    recent_agent_rewards_at_death = collections.deque(maxlen=20)
    recent_agent_lifespans = collections.deque(maxlen=20)

    training_queue = queue.Queue()
    training_thread = threading.Thread(target=training_worker, args=(ppo_agent, training_queue))
    training_thread.daemon = True
    training_thread.start()
    print("Training thread started.")

    tick = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                renderer.quit()
                training_queue.put(None)
                training_thread.join()
                print("Training thread stopped.")
                return

        actions_for_env = {}
        current_values = {}
        current_log_probs = {}
        main_agent_action_outputs = None

        for agent_id in env.agents.keys():
            if agent_id not in agent_hidden_states:
                agent_hidden_states[agent_id] = ppo_agent.policy_net.init_hidden()
            agent_lifespans[agent_id] += 1

        observations = {agent_id: agent.get_sight_input(env) for agent_id, agent in env.agents.items() if agent.is_alive}

        for agent_id, obs in observations.items():
            # BUG FIX: Reshape obs_tensor to be 3D [batch, seq, features] for the RNN
            obs_tensor = torch.tensor(obs, dtype=torch.float32).view(1, 1, -1)

            with torch.no_grad():
                binary_logits, continuous_means, log_std, value, next_hidden_state = ppo_agent.policy_net(obs_tensor, agent_hidden_states[agent_id])

            current_values[agent_id] = value.item()
            agent_hidden_states[agent_id] = next_hidden_state

            # --- Sample Actions from Distributions ---
            # Binary actions
            binary_dist = Bernoulli(logits=binary_logits.squeeze(0))
            binary_actions_sampled = binary_dist.sample()
            
            # Continuous actions
            std = torch.exp(log_std)
            continuous_dist = Normal(continuous_means.squeeze(0), std)
            continuous_actions_sampled = continuous_dist.sample().clamp(0, 1) # Clamp to valid range [0, 1]
            
            # Combine actions
            turn_clockwise_sampled, turn_counterclockwise_sampled, move_forward_sampled, eat_sampled, attack_sampled = binary_actions_sampled.flatten().tolist()
            turn_speed_sampled, movement_speed_sampled = continuous_actions_sampled.flatten().tolist()

            sampled_actions_tensor = torch.tensor([
                turn_clockwise_sampled, turn_counterclockwise_sampled,
                turn_speed_sampled,
                move_forward_sampled,
                movement_speed_sampled,
                eat_sampled, attack_sampled
            ], dtype=torch.float32)

            actions_for_env[agent_id] = sampled_actions_tensor

            if agent_id == env.main_agent_id:
                main_agent_action_outputs = sampled_actions_tensor.tolist()

            # Calculate log probabilities for PPO of the sampled actions
            log_prob_binary = binary_dist.log_prob(binary_actions_sampled).sum()
            log_prob_continuous = continuous_dist.log_prob(continuous_actions_sampled).sum()
            current_log_probs[agent_id] = (log_prob_binary + log_prob_continuous).item()

        next_observations, rewards, done, info, next_hp_states = env.step(actions_for_env)

        agents_to_train_this_tick = []
        for agent_id in list(observations.keys()):
            is_dead = agent_id not in env.agents
            
            # Store experience from this tick
            all_experiences[agent_id]['observations'].append(observations[agent_id])
            all_experiences[agent_id]['actions'].append(actions_for_env[agent_id].numpy())
            all_experiences[agent_id]['rewards'].append(rewards.get(agent_id, 0))
            all_experiences[agent_id]['values'].append(current_values[agent_id])
            all_experiences[agent_id]['log_probs'].append(current_log_probs[agent_id])
            all_experiences[agent_id]['dones'].append(1 if is_dead else 0)
            all_experiences[agent_id]['hidden_states'].append(agent_hidden_states[agent_id].squeeze(0).numpy())

            if is_dead:
                all_experiences[agent_id]['values'].append(0.0) # Final value is 0 for dead agents
                
                total_reward_this_life = sum(all_experiences[agent_id]['rewards'])
                recent_agent_rewards_at_death.append(total_reward_this_life)
                recent_agent_lifespans.append(agent_lifespans[agent_id])
                del agent_lifespans[agent_id]
                
                agents_to_train_this_tick.append(agent_id)
                if agent_id in agent_hidden_states:
                    del agent_hidden_states[agent_id]

        all_agents_to_consider_for_training = set(agents_to_train_this_tick)
        for agent_id in list(all_experiences.keys()):
            if agent_id not in agents_to_train_this_tick and len(all_experiences[agent_id]['rewards']) >= SimulationConfig.N_STEPS:
                all_agents_to_consider_for_training.add(agent_id)

        experiences_to_train_batch = {}
        for agent_id in all_agents_to_consider_for_training:
            if agent_id in all_experiences:
                if agent_id not in agents_to_train_this_tick: # Agent is alive, so bootstrap value
                    if agent_id in observations and agent_id in agent_hidden_states:
                        final_obs_tensor = torch.tensor(observations[agent_id], dtype=torch.float32).view(1, 1, -1)
                        final_hidden_state = agent_hidden_states[agent_id]
                        with torch.no_grad():
                            _, _, _, final_value, _ = ppo_agent.policy_net(final_obs_tensor, final_hidden_state)
                        all_experiences[agent_id]['values'].append(final_value.item())
                    else:
                        all_experiences[agent_id]['values'].append(0.0)

                experiences_to_train_batch[agent_id] = all_experiences.pop(agent_id)

        if experiences_to_train_batch:
            training_queue.put(experiences_to_train_batch)

        main_agent_data = env.get_main_agent_data()
        if main_agent_data:
            main_agent_data["action_outputs"] = main_agent_action_outputs
            if env.main_agent_id in all_experiences:
                main_agent_data["current_cumulative_reward"] = sum(all_experiences[env.main_agent_id]['rewards'])
            else:
                main_agent_data["current_cumulative_reward"] = 0

        renderer.draw(env.get_drawable_entities(), tick, main_agent_data,
                      recent_agent_rewards_at_death, recent_agent_lifespans)
        renderer.tick()
        tick += 1

if __name__ == "__main__":
    run_simulation()