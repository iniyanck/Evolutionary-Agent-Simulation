# training.py

import torch
import torch.optim as optim
import numpy as np
from config import SimulationConfig
from network import AgentRNN
import torch.nn.functional as F
from torch.distributions import Bernoulli, Normal

class PPO:
    def __init__(self, input_dim, output_dim):
        self.policy_net = AgentRNN(input_dim, output_dim, SimulationConfig.RNN_HIDDEN_SIZE)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=SimulationConfig.LEARNING_RATE)
        self.gamma = SimulationConfig.GAMMA
        self.gae_lambda = SimulationConfig.GAE_LAMBDA
        self.ppo_epsilon = SimulationConfig.PPO_EPSILON
        self.ppo_epochs = SimulationConfig.PPO_EPOCHS
        self.entropy_coefficient = SimulationConfig.ENTROPY_COEFFICIENT

    def compute_gae(self, rewards, values, dones):
        values = np.array(values, dtype=np.float32)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae_lam = 0
        num_steps = len(rewards)
        
        for t in reversed(range(num_steps)):
            next_non_terminal = 1.0 - dones[t]
            next_value = values[t+1] # values has N+1 elements
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam
        
        returns = advantages + values[:-1]
        return torch.tensor(advantages, dtype=torch.float32), torch.tensor(returns, dtype=torch.float32)

    def train(self, experiences):
        if not experiences:
            return 0.0

        loss_sum = 0
        num_agents = 0

        for agent_id, agent_exp in experiences.items():
            if not agent_exp['rewards']:
                continue
            
            num_agents += 1
            obs_batch = torch.tensor(np.array(agent_exp['observations']), dtype=torch.float32).unsqueeze(0)
            actions_batch = torch.tensor(np.array(agent_exp['actions']), dtype=torch.float32)
            old_log_probs_batch = torch.tensor(np.array(agent_exp['log_probs']), dtype=torch.float32)
            
            # The first hidden state in the list is the initial one for the sequence
            initial_hidden_state = torch.tensor(np.array(agent_exp['hidden_states'][0]), dtype=torch.float32).unsqueeze(0)

            advantages, returns = self.compute_gae(
                agent_exp['rewards'], agent_exp['values'], agent_exp['dones']
            )
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            for _ in range(self.ppo_epochs):
                binary_logits, continuous_means, log_std, new_values, _ = self.policy_net(obs_batch, initial_hidden_state)
                
                new_values = new_values.squeeze(0).squeeze(-1)
                binary_logits = binary_logits.squeeze(0)
                continuous_means = continuous_means.squeeze(0)
                
                # Re-calculate log probs and entropy for all actions
                binary_actions = actions_batch[:, [0, 1, 3, 5, 6]]
                continuous_actions = actions_batch[:, [2, 4]]

                binary_dist = Bernoulli(logits=binary_logits)
                log_prob_binary = binary_dist.log_prob(binary_actions).sum(dim=-1)
                entropy_binary = binary_dist.entropy().sum(dim=-1)

                std = torch.exp(log_std)
                continuous_dist = Normal(continuous_means, std)
                log_prob_continuous = continuous_dist.log_prob(continuous_actions).sum(dim=-1)
                entropy_continuous = continuous_dist.entropy().sum(dim=-1)
                
                new_log_probs = log_prob_binary + log_prob_continuous
                entropy = (entropy_binary + entropy_continuous).mean()
                
                ratio = torch.exp(new_log_probs - old_log_probs_batch)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.ppo_epsilon, 1.0 + self.ppo_epsilon) * advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(new_values, returns)
                
                loss = policy_loss + 0.5 * value_loss - self.entropy_coefficient * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
                self.optimizer.step()
            
            loss_sum += loss.item()

        return loss_sum / num_agents if num_agents > 0 else 0.0