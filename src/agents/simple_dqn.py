import numpy as np # <-- HATA DÜZELTMESİ: Eksik olan import satırı eklendi.
import torch
import torch.nn as nn
import random
from collections import deque, namedtuple

class QNetwork(nn.Module):
    """
    Durumu girdi olarak alıp, her eylem için Q-değerlerini tahmin eden ağ.
    """
    def __init__(self, obs_space_size, action_space_size):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_space_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_space_size) # Softmax YOK! Doğrudan Q-değerlerini tahmin ediyoruz.
        )

    def forward(self, state):
        return self.net(state)

class ReplayBuffer:
    """
    Deneyimleri (s, a, r, s', done) saklamak için bir hafıza yapısı.
    """
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Hafızaya yeni bir deneyim ekler."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self, batch_size):
        """Hafızadan rastgele bir batch örnekler."""
        return random.sample(self.memory, k=batch_size)
    
    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, obs_space_size, action_space_size, device, buffer_size=10000, batch_size=64, lr=1e-3, gamma=0.99, tau=1e-3, update_every=4):
        self.action_space_size = action_space_size
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau # Hedef ağın yavaşça güncellenmesi için
        self.update_every = update_every
        
        self.q_network = QNetwork(obs_space_size, action_space_size).to(device)
        self.target_network = QNetwork(obs_space_size, action_space_size).to(device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        
        self.memory = ReplayBuffer(buffer_size)
        self.time_step = 0

    def step(self, state, action, reward, next_state, done):
        # Deneyimi hafızaya kaydet
        self.memory.add(state, action, reward, next_state, done)
        
        # Belirli aralıklarla öğrenme adımını tetikle
        self.time_step = (self.time_step + 1) % self.update_every
        if self.time_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size)
                self.learn(experiences)

    def select_action(self, state, epsilon=0.0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.q_network.eval() # Ağı değerlendirme moduna al
        with torch.no_grad():
            action_values = self.q_network(state)
        self.q_network.train() # Ağı tekrar eğitim moduna al
        
        # Epsilon-Greedy eylem seçimi
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_space_size))

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        states = torch.from_numpy(np.vstack(states)).float().to(self.device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(np.vstack(next_states))).float().to(self.device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)

        # 1. Hedef Ağı ile bir sonraki durumların max Q-değerlerini hesapla
        q_targets_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
        
        # 2. TD Hedefini Hesapla: R + gamma * Q_target(s', a')
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
        
        # 3. Ana Ağdan mevcut tahminleri al
        q_expected = self.q_network(states).gather(1, actions)
        
        # 4. Kaybı Hesapla (MSE Loss)
        loss = nn.MSELoss()(q_expected, q_targets)
        
        # 5. Ağı Güncelle
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 6. Hedef Ağı Yavaşça Güncelle (Soft Update)
        self.soft_update(self.q_network, self.target_network)

    def soft_update(self, local_model, target_model):
        """Hedef ağın ağırlıklarını yavaşça ana ağın ağırlıklarına yaklaştır."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
