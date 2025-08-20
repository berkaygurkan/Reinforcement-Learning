# src/agents/simple_pg.py

import torch
import torch.nn as nn
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    """
    Durumu girdi olarak alıp, her eylem için bir olasılık üreten
    basit bir politika ağı.
    """
    def __init__(self, obs_space_size, action_space_size):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_space_size, 128),
            nn.ReLU(), # Nonlineerite katmak için
            nn.Linear(128, action_space_size),
            nn.Softmax(dim=-1) # Olasılık dağılımı elde etmek için Softmax. Böylelikle toplamları 1 olacak
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
        self.gamma = gamma # İndirgeme faktörü
        self.device = device
        
        # Bölüm boyunca toplanan ödülleri ve log-olasılıkları saklamak için listeler
        self.rewards = []
        self.log_probs = []

    def select_action(self, state):
        """
        Verilen duruma göre bir eylem seçer.
        """
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        # Politika ağından eylem olasılıklarını al
        action_probs = self.policy_network(state_tensor)
        
        # Bu olasılık dağılımından bir eylem örnekle
        #Categorical(action_probs): Politika ağından gelen [0.7, 0.3] gibi olasılıkları alır 
        # ve ondan örneklem (sampling) yapabileceğimiz bir olasılık dağılım nesnesi oluşturur. 
        # dist.sample() komutu, bu olasılıklara göre bir eylem seçer (%70 ihtimalle birini, 
        # %30 ihtimalle diğerini). Bu, stokastik politikanın hayata geçtiği yerdir.
        dist = Categorical(action_probs)
        action = dist.sample()
        
        # Öğrenme adımı için bu eylemin log-olasılığını kaydet
        self.log_probs.append(dist.log_prob(action))
        
        return action.item()

    def update_policy(self):
        """
        Bir bölüm tamamlandıktan sonra politika ağını günceller.
        """
        discounted_rewards = []
        cumulative_reward = 0
        
        # Ödülleri sondan başa doğru indirgeyerek hesapla (Gt)
        for reward in reversed(self.rewards):
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        
        # Tensöre çevir ve normalize et (varyansı azaltmak için önemli bir adım)
        discounted_rewards = torch.tensor(discounted_rewards, device=self.device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        
        policy_loss = []
        for log_prob, reward in zip(self.log_probs, discounted_rewards):
            # Kayıp fonksiyonu: -log(pi(a|s)) * Gt
            # Gradyan yükselmesi (ascent) yapmak yerine, PyTorch'un gradyan
            # alçalması (descent) yapısını kullanmak için kayıp fonksiyonunu negatif yaparız.
            policy_loss.append(-log_prob * reward)
            
        # Gradyanları sıfırla
        self.optimizer.zero_grad()
        
        # Kaybı topla, geriye yayılım yap ve ağırlıkları güncelle
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        # Bir sonraki bölüm için hafızayı temizle
        self.rewards = []
        self.log_probs = []
