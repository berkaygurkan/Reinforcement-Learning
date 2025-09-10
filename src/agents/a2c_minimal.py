import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

class ActorCriticNetwork(nn.Module):
    """
    Hem politika (Aktör) hem de değer (Kritik) tahminini yapan birleşik ağ.
    """
    def __init__(self, obs_space_size, action_space_size):
        super(ActorCriticNetwork, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(obs_space_size, 128),
            nn.ReLU()
        )
        self.policy_head = nn.Sequential(
            nn.Linear(128, action_space_size),
            nn.Softmax(dim=-1)
        )
        self.value_head = nn.Linear(128, 1)

    def forward(self, state):
        x = self.shared_layers(state)
        action_probs = self.policy_head(x)
        state_value = self.value_head(x)
        return action_probs, state_value

class A2CAgent:
    def __init__(self, obs_space_size, action_space_size, device, lr=7e-4, gamma=0.99, gae_lambda=0.95, entropy_coef=0.01, value_loss_coef=0.5):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        
        self.network = ActorCriticNetwork(obs_space_size, action_space_size).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action_probs, state_value = self.network(state)
        
        dist = Categorical(action_probs)
        action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob, state_value

    def update_policy(self, rollout):
        states, actions, rewards, log_probs, values, dones, next_value = rollout

        advantages = torch.zeros_like(rewards).to(self.device)
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            # --- HATA DÜZELTMESİ BURADA ---
            # 'dones' tensörünü, çıkarma işlemi yapmadan önce float'a çeviriyoruz.
            mask = 1.0 - dones[t].float()
            
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            last_advantage = delta + self.gamma * self.gae_lambda * last_advantage * mask
            advantages[t] = last_advantage
            next_value = values[t]
        
        returns = advantages + values

        policy_loss = -(advantages.detach() * log_probs).mean()
        value_loss = nn.MSELoss()(returns.squeeze(), values.squeeze())
        
        dist = Categorical(self.network(states)[0])
        entropy_loss = dist.entropy().mean()

        total_loss = policy_loss - self.entropy_coef * entropy_loss + self.value_loss_coef * value_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

