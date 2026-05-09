# %% [1] KURULUM VE KÜTÜPHANELER
import os
import random
import dataclasses
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    raise ImportError("Lütfen 'sb3-contrib' kurun: pip install sb3-contrib")

# Donanım Seçimi
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[SİSTEM] Çalışma birimi: {DEVICE}")

# Tekrarlanabilirlik için Seed
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# %% [2] KONFİGÜRASYON
@dataclasses.dataclass
class TrainingConfig:
    experiment_name: str = "Reacher_FrictionLoss_Training"
    env_id: str = "Reacher-v4"
    seed: int = 42
    
    # Eğitim Süresi (Her model için)
    total_timesteps: int = 2_000_000  # 3 Milyon adım (Friction zor olduğu için artırılabilir)
    n_envs: int = 16                   # Paralel ortam sayısı
    
    # Sürtünme Aralığı (Eğitim Sırasında)
    # Reacher'da varsayılan frictionloss 0'dır.
    # Biz bunu 0.0 (yok) ile 2.0 (sert sürtünme) arasında rastgele değiştireceğiz.
    friction_min: float = 0.0
    friction_max: float = 2.0
    
    result_dir: str = "./results_friction_training"

CFG = TrainingConfig()
seed_everything(CFG.seed)
os.makedirs(CFG.result_dir, exist_ok=True)

# %% [3] FRICTION TRAINING WRAPPER
class FrictionTrainingWrapper(gym.Wrapper):
    def __init__(self, env, dr_mode: bool = False):
        """
        dr_mode=True: Her epizot başında frictionloss rastgele atanır (DR ve LSTM için).
        dr_mode=False: Frictionloss hep 0 kalır (Vanilla için).
        """
        super().__init__(env)
        self.dr_mode = dr_mode
        
        # Gözlem uzayını genişlet (LSTM için gerekli: Obs + Action + Reward + Done)
        self.act_dim = env.action_space.shape[0]
        self.obs_dim = env.observation_space.shape[0]
        self.augmented_dim = self.obs_dim + self.act_dim + 2 
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.augmented_dim,), dtype=np.float32
        )
        
        # Hafıza değişkenleri
        self.prev_action = np.zeros(self.act_dim, dtype=np.float32)
        self.prev_reward = 0.0
        self.prev_done = 0.0
        
        # Orijinal değerleri saklamaya gerek yok çünkü her reset'te üzerine yazacağız.

    def reset(self, **kwargs):
        self.prev_action = np.zeros(self.act_dim, dtype=np.float32)
        self.prev_reward = 0.0
        self.prev_done = 0.0
        
        # --- FRICTION LOSS AYARLAMASI ---
        if hasattr(self.unwrapped, "model"):
            if self.dr_mode:
                # Domain Randomization: 0.0 ile 2.0 arasında rastgele kuru sürtünme
                friction_val = np.random.uniform(CFG.friction_min, CFG.friction_max)
                self.unwrapped.model.dof_frictionloss[:] = friction_val
            else:
                # Vanilla: Sürtünme yok (Varsayılan)
                self.unwrapped.model.dof_frictionloss[:] = 0.0
        
        obs, info = self.env.reset(**kwargs)
        return self._augment_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        # LSTM için geçmiş bilgileri kaydet
        self.prev_action = action
        self.prev_reward = float(reward)
        self.prev_done = float(done)
        
        return self._augment_obs(obs), reward, terminated, truncated, info

    def _augment_obs(self, obs):
        return np.concatenate([
            obs, self.prev_action, [self.prev_reward], [self.prev_done]
        ]).astype(np.float32)

# %% [4] ORTAM OLUŞTURUCU (FACTORY)
def make_env(rank: int, dr_mode: bool = False, log_dir: str = None):
    def _init():
        env = gym.make(CFG.env_id)
        # Sürtünme Wrapper'ını ekle
        env = FrictionTrainingWrapper(env, dr_mode=dr_mode)
        if log_dir:
            env = Monitor(env, os.path.join(log_dir, str(rank)))
        return env
    return _init

# %% [5] EĞİTİM YÖNETİCİSİ
def train_agent(algo_name: str):
    """
    algo_name: 'vanilla' | 'dr' | 'lstm'
    """
    print(f"\n{'='*40}")
    print(f"EĞİTİM BAŞLIYOR: {algo_name.upper()}")
    print(f"Hedef: 'dof_frictionloss' adaptasyonu")
    print(f"{'='*40}")
    
    run_name = f"{algo_name}_seed{CFG.seed}"
    log_dir = os.path.join(CFG.result_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    
    model_path = os.path.join(CFG.result_dir, f"{algo_name}_model.zip")
    stats_path = os.path.join(CFG.result_dir, f"{algo_name}_vecnorm.pkl")
    
    if os.path.exists(model_path):
        print(f"[BİLGİ] {algo_name} zaten eğitilmiş, atlanıyor.")
        return

    # --- DR MODE BELİRLEME ---
    # Vanilla: dr_mode=False (Sabit 0 sürtünme ile eğitilir)
    # DR ve LSTM: dr_mode=True (0.0 - 2.0 arası rastgele sürtünme ile eğitilir)
    is_dr = (algo_name in ['dr', 'lstm'])
    
    # Ortamları Oluştur
    envs = SubprocVecEnv([
        make_env(rank=i, dr_mode=is_dr, log_dir=log_dir) 
        for i in range(CFG.n_envs)
    ])
    
    # Normalizasyon (Eğitim stabilitesi için şart)
    envs = VecNormalize(envs, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # Model Seçimi
    if algo_name == 'lstm':
        # RecurrentPPO (RL^2 Mimarisi)
        # LSTM layerları geçmişteki sürtünme etkisini "hatırlayacak"
        policy_kwargs = {
            "lstm_hidden_size": 256,
            "n_lstm_layers": 2,
            "shared_lstm": True,
            "enable_critic_lstm": False
        }
        model = RecurrentPPO(
            "MlpLstmPolicy", 
            envs, 
            verbose=1, 
            device=DEVICE,
            batch_size=512, 
            n_steps=2048, 
            learning_rate=3e-4, 
            ent_coef=0.01, # Biraz keşif teşviği
            policy_kwargs=policy_kwargs
        )
    else:
        # Vanilla ve DR için Standart PPO (MlpPolicy)
        # Not: Wrapper sayesinde input boyutu LSTM ile aynıdır (adil karşılaştırma)
        model = PPO(
            "MlpPolicy", 
            envs, 
            verbose=1, 
            device=DEVICE,
            batch_size=512, 
            n_steps=2048, 
            learning_rate=3e-4,
            ent_coef=0.01
        )
        
    # Eğitimi Başlat
    model.learn(total_timesteps=CFG.total_timesteps)
    
    # Kaydet
    model.save(model_path)
    envs.save(stats_path)
    envs.close()
    print(f"[TAMAMLANDI] {algo_name} kaydedildi.")

# %% [6] ANA ÇALIŞTIRMA BLOĞU
if __name__ == "__main__":
    
    # 1. Vanilla PPO Eğit (Sabit Ortam)
    train_agent("vanilla")
    
    # 2. Domain Randomization PPO Eğit (Rastgele Sürtünme, Hafızasız)
    train_agent("dr")
    
    # 3. RL^2 LSTM Eğit (Rastgele Sürtünme, Hafızalı)
    train_agent("lstm")
    
    print("\n>>> TÜM EĞİTİMLER TAMAMLANDI.")
    print(f">>> Modeller '{CFG.result_dir}' klasörüne kaydedildi.")