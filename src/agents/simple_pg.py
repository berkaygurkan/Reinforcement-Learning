import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

class PolicyNetwork(nn.Module):
    """
    Durumu girdi olarak alıp eylemler üzerinde bir olasılık dağılımı döndüren ağ.
    """
    def __init__(self, obs_space_size, action_space_size):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_space_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_space_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        return self.net(state)

class ReinforceAgent:
    """
    REINFORCE algoritmasını uygulayan ajan.
    """
    def __init__(self, obs_space_size, action_space_size, device, lr=1e-3, gamma=0.99):
        self.policy_network = PolicyNetwork(obs_space_size, action_space_size).to(device)
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=lr)
        self.gamma = gamma
        self.device = device
        # Bölüm içi hafıza
        self.rewards = []
        self.log_probs = []

    def select_action(self, state):
        """Mevcut politikaya göre bir eylem seçer ve log-olasılığını kaydeder."""
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action_probs = self.policy_network(state_tensor)
        dist = Categorical(action_probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def update_policy(self, use_baseline=True):
        """
        Bir bölüm sonunda toplanan verilere göre politika ağını günceller.
        
        Args:
            use_baseline (bool): Eğer True ise, varyansı azaltmak için ödülleri normalize eder.
        """
        # Adım 1: İndirgenmiş Getirileri (Discounted Returns) Hesapla
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(self.rewards):
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
            
        discounted_rewards = torch.tensor(discounted_rewards, device=self.device)
        
        # --- KRİTİK GÜNCELLEME ---
        # Eğer baseline kullanılıyorsa, ödülleri normalize et.
        if use_baseline and len(discounted_rewards) > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        # Adım 3: Kayıp Fonksiyonunu Oluştur
        policy_loss = []
        for log_prob, reward in zip(self.log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
            
        # Adım 4: Ağırlıkları Güncelle
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        # Hafızayı bir sonraki bölüm için temizle
        self.rewards = []
        self.log_probs = []

