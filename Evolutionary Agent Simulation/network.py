# network.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import SimulationConfig

class AgentRNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size):
        super(AgentRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        
        # Policy head for binary actions (logits)
        self.fc_policy_binary = nn.Linear(hidden_size, 5) # 5 binary actions

        # Policy head for continuous actions (mean values)
        self.fc_policy_continuous_mean = nn.Linear(hidden_size, 2) # 2 continuous actions
        
        # Learnable log standard deviation for continuous actions
        self.log_std = nn.Parameter(torch.zeros(2))

        # Value head
        self.fc_value = nn.Linear(hidden_size, 1)

    def forward(self, x, h_in):
        # x shape: (batch_size, sequence_length, input_dim)
        # h_in shape: (num_layers, batch_size, hidden_size)
        x = F.relu(self.fc1(x))
        rnn_out, h_out = self.rnn(x, h_in.contiguous())

        # --- Policy Outputs ---
        # Get logits for the 5 binary actions
        binary_logits = self.fc_policy_binary(rnn_out)
        
        # Get means for the 2 continuous actions and apply sigmoid to bound them in [0, 1]
        continuous_means = torch.sigmoid(self.fc_policy_continuous_mean(rnn_out))

        # --- Value Output ---
        value = self.fc_value(rnn_out)

        return binary_logits, continuous_means, self.log_std, value, h_out

    def init_hidden(self, batch_size=1):
        # Returns a hidden state tensor compatible with the GRU layer
        return torch.zeros(1, batch_size, self.hidden_size, device='cpu')