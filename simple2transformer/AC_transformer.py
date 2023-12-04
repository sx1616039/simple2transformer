import torch.nn as nn
import torch.nn.functional as F

from transformer import Transformer


class Actor(nn.Module):
    def __init__(self, d_model, num_output=6):
        super(Actor, self).__init__()
        self.d_model = d_model
        self.transformer_encoder = Transformer(d_model=d_model)
        self.action_head = nn.Linear(d_model, num_output)

    def forward(self, x):
        x = self.transformer_encoder(x)
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self, d_model, num_output=1):
        super(Critic, self).__init__()

        self.d_model = d_model
        self.transformer_encoder = Transformer(d_model=d_model)
        self.state_value = nn.Linear(d_model, num_output)

    def forward(self, x):
        x = self.transformer_encoder(x)
        value = self.state_value(x)
        return value

