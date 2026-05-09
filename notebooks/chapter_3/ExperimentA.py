# %% [1] SETUP & LIBRARIES
import os
import glob
import time
import json
import random
import dataclasses
import numpy as np
import pandas as pd
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Any, Dict, Tuple, List
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

# RecurrentPPO (LSTM) Kontrolü
try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    raise ImportError("Lütfen 'sb3-contrib' kurun: pip install sb3-contrib")

# --- DONANIM YÖNETİMİ ---
def get_device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps" # Mac Silicon
    return "cpu"

DEVICE = get_device()
print(f"[SYSTEM] Running on: {DEVICE}")

# --- TEKRARLANABİLİRLİK (SEEDING) ---
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# %% [2] CONFIGURATION
@dataclasses.dataclass
class ExperimentConfig:
    experiment_name: str = "MetaReacher_Thesis_Final"
    env_id: str = "Reacher-v5" # Gymnasium'da v5 henüz stabil olmayabilir, v4 önerilir. v5 istiyorsanız burayı "Reacher-v5" yapın.
    seed: int = 42
    
    # Eğitim Parametreleri
    total_timesteps: int = 200_000 # Hızlı test için 200k, Tez için 1M yapın
    n_envs: int = 8                # Paralel ortam sayısı
    
    # Meta-RL Senaryosu
    mass_change_step: int = 25
    mass_multiplier: float = 3.0
    
    # Kayıt Yolları
    result_dir: str = "./results"

CFG = ExperimentConfig()
seed_everything(CFG.seed)
os.makedirs(CFG.result_dir, exist_ok=True)

# %% [3] THE META-WRAPPER
class MetaReacherWrapper(gym.Wrapper):
    def __init__(self, env, is_training: bool = True, dr_mode: bool = False, 
                 dynamic_test_mode: bool = False):
        super().__init__(env)
        self.is_training = is_training
        self.dr_mode = dr_mode
        self.dynamic_test_mode = dynamic_test_mode
        
        self.act_dim = env.action_space.shape[0]
        self.obs_dim = env.observation_space.shape[0]
        self.augmented_dim = self.obs_dim + self.act_dim + 2 
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.augmented_dim,), dtype=np.float32
        )
        
        self.prev_action = np.zeros(self.act_dim, dtype=np.float32)
        self.prev_reward = 0.0
        self.prev_done = 0.0
        self.step_count = 0
        
        # Physics Access (MuJoCo)
        # Reacher-v4 ve v5 uyumluluğu için
        if hasattr(self.unwrapped, "model"):
            self.original_masses = self.unwrapped.model.body_mass.copy()
        else:
            print("[UYARI] MuJoCo modeli bulunamadı, kütle değişimi çalışmayabilir.")
            self.original_masses = None

    def reset(self, **kwargs):
        self.step_count = 0
        self.prev_action = np.zeros(self.act_dim, dtype=np.float32)
        self.prev_reward = 0.0
        self.prev_done = 0.0
        
        scale = 1.0
        if self.is_training and self.dr_mode:
            scale = np.random.uniform(0.5, 2.0)
            
        self._set_mass_scale(scale)
        obs, info = self.env.reset(**kwargs)
        return self._augment_obs(obs), info

    def step(self, action):
        self.step_count += 1
        
        # Test sırasında arıza enjeksiyonu
        if self.dynamic_test_mode and self.step_count == CFG.mass_change_step:
            self._set_mass_scale(CFG.mass_multiplier)

        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        aug_obs = self._augment_obs(obs)
        self.prev_action = action
        self.prev_reward = float(reward)
        self.prev_done = float(done)
        
        return aug_obs, reward, terminated, truncated, info

    def _augment_obs(self, obs):
        return np.concatenate([
            obs, self.prev_action, [self.prev_reward], [self.prev_done]
        ]).astype(np.float32)

    def _set_mass_scale(self, scale: float):
        if self.original_masses is not None:
            new_mass = self.original_masses.copy()
            # Link 2 genellikle uç eyleyicidir
            new_mass[2] *= scale 
            self.unwrapped.model.body_mass[:] = new_mass

# %% [4] FACTORY FUNCTIONS
def make_env(rank: int, is_training=True, dr_mode=False, dynamic_test_mode=False, log_dir=None):
    def _init():
        env = gym.make(CFG.env_id)
        env = MetaReacherWrapper(env, is_training, dr_mode, dynamic_test_mode)
        
        if log_dir:
            log_file = os.path.join(log_dir, str(rank))
            env = Monitor(env, log_file)
        return env
    return _init

# %% [5] TRAINING ENGINE
def train_manager(algo_type: str):
    run_name = f"{algo_type}_seed{CFG.seed}"
    log_dir = os.path.join(CFG.result_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    
    model_path = os.path.join(CFG.result_dir, f"{algo_type}_model.zip")
    stats_path = os.path.join(CFG.result_dir, f"{algo_type}_vecnorm.pkl")

    if os.path.exists(model_path) and os.path.exists(stats_path):
        print(f"[SKIP] {algo_type} zaten eğitilmiş.")
        return

    print(f"\n>>> Eğitim Başlıyor: {algo_type.upper()}")
    is_dr = (algo_type in ['dr', 'lstm'])
    
    envs = SubprocVecEnv([
        make_env(rank=i, is_training=True, dr_mode=is_dr, log_dir=log_dir) 
        for i in range(CFG.n_envs)
    ])
    
    envs = VecNormalize(envs, norm_obs=True, norm_reward=True, clip_obs=10.)

    if algo_type == 'lstm':
        policy_kwargs = {
            "lstm_hidden_size": 256,
            "n_lstm_layers": 2,
            "shared_lstm": True,
            "enable_critic_lstm": False
        }
        model = RecurrentPPO(
            "MlpLstmPolicy", envs, verbose=1, device=DEVICE,
            batch_size=512, n_steps=2048, learning_rate=3e-4, ent_coef=0.01,
            policy_kwargs=policy_kwargs
        )
    else:
        model = PPO(
            "MlpPolicy", envs, verbose=1, device=DEVICE,
            batch_size=512, n_steps=2048, learning_rate=3e-4, ent_coef=0.01
        )

    model.learn(total_timesteps=CFG.total_timesteps)
    model.save(model_path)
    envs.save(stats_path)
    print(f"[SAVED] {algo_type} kaydedildi.")
    
    envs.close()
    del model

# %% [6] VISUALIZATION ENGINE (FIXED)
def plot_comprehensive_results():
    print("\n>>> Analiz ve Görselleştirme Başlıyor...")
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    palette = {"Vanilla PPO": "#e74c3c", "DR-PPO": "#f1c40f", "RL²-LSTM": "#2ecc71"}
    
    models_config = {
        "Vanilla PPO": {"file": "vanilla", "class": PPO},
        "DR-PPO":      {"file": "dr",      "class": PPO},
        "RL²-LSTM":    {"file": "lstm",    "class": RecurrentPPO},
    }
    
    df_training = pd.DataFrame()
    list_standard = []
    list_ood = []
    list_adaptation = []

    # 1. Eğitim Loglarını Oku
    for name, conf in models_config.items():
        log_pattern = os.path.join(CFG.result_dir, f"{conf['file']}_seed{CFG.seed}", "*.monitor.csv")
        monitor_files = glob.glob(log_pattern)
        
        for file in monitor_files:
            try:
                df = pd.read_csv(file, skiprows=1)
                if 'l' in df.columns and 'r' in df.columns:
                    df['Algorithm'] = name
                    df['Timesteps'] = df['l'].cumsum()
                    df['Reward'] = df['r'].ewm(alpha=0.05).mean()
                    df_training = pd.concat([df_training, df])
            except Exception as e:
                print(f"Log okuma hatası ({name}): {e}")

    # 2. Test Döngüsü
    for name, conf in models_config.items():
        model_path = os.path.join(CFG.result_dir, f"{conf['file']}_model.zip")
        stats_path = os.path.join(CFG.result_dir, f"{conf['file']}_vecnorm.pkl")
        
        if not os.path.exists(model_path): 
            print(f"[ATLANDI] Model bulunamadı: {name}")
            continue
        
        print(f"Testing {name}...")
        # CPU kullanarak yükle (GPU uyarısını engeller)
        agent = conf['class'].load(model_path, device="cpu")
        
        # A. Standart & OOD Testi
        for mass_scale, target_list in [(1.0, list_standard), (3.0, list_ood)]:
            env = DummyVecEnv([make_env(rank=0, is_training=False, dr_mode=False)])
            
            if os.path.exists(stats_path):
                env = VecNormalize.load(stats_path, env)
                env.training = False 
                env.norm_reward = False
            
            # --- CRITICAL FIX START ---
            # .unwrapped yerine get_wrapper_attr kullanıyoruz.
            # Bu, iç içe geçmiş wrapper'lar arasında 'MetaReacherWrapper'daki fonksiyonu bulur.
            try:
                env.envs[0].get_wrapper_attr('_set_mass_scale')(mass_scale)
            except AttributeError:
                # Fallback: Eğer eski gym versiyonu ise elle ara
                curr = env.envs[0]
                found = False
                while hasattr(curr, 'env'):
                    if hasattr(curr, '_set_mass_scale'):
                        curr._set_mass_scale(mass_scale)
                        found = True
                        break
                    curr = curr.env
                if not found and hasattr(curr, '_set_mass_scale'):
                     curr._set_mass_scale(mass_scale)
            # --- CRITICAL FIX END ---
            
            for _ in range(10): # 10 Epizot
                obs = env.reset()
                done = False
                ep_rew = 0
                lstm_states = None
                episode_starts = np.ones((1,), dtype=bool)
                
                while not done:
                    if conf['class'] == RecurrentPPO:
                        action, lstm_states = agent.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
                    else:
                        action, _ = agent.predict(obs, deterministic=True)
                    
                    obs, reward, done_arr, _ = env.step(action)
                    ep_rew += reward[0]
                    done = done_arr[0]
                    episode_starts = done_arr
                
                target_list.append({"Algorithm": name, "Reward": ep_rew})
            env.close()

        # B. Adaptasyon Testi
        env = DummyVecEnv([make_env(rank=0, is_training=False, dynamic_test_mode=True)])
        if os.path.exists(stats_path):
            env = VecNormalize.load(stats_path, env)
            env.training = False
            env.norm_reward = False
        
        for ep in range(5): # 5 Epizot
            obs = env.reset()
            lstm_states = None
            episode_starts = np.ones((1,), dtype=bool)
            
            for t in range(50):
                if conf['class'] == RecurrentPPO:
                    action, lstm_states = agent.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
                else:
                    action, _ = agent.predict(obs, deterministic=True)
                
                obs, reward, done, info = env.step(action)
                
                # Reacher reward_dist verisi (hedefe uzaklık)
                dist_error = info[0].get('reward_dist', -reward[0])
                
                list_adaptation.append({
                    "Algorithm": name, "Step": t, 
                    "Tracking Error": abs(dist_error), "Episode": ep
                })
                episode_starts = done
                if done[0]: break
        env.close()

    # 3. Çizim
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    if not df_training.empty:
        sns.lineplot(data=df_training, x="Timesteps", y="Reward", hue="Algorithm", palette=palette, ax=axes[0,0])
        axes[0,0].set_title("1. Training Curve")
    else:
        axes[0,0].text(0.5, 0.5, "Log Files Not Found", ha='center')
    
    if list_standard:
        sns.barplot(data=pd.DataFrame(list_standard), x="Algorithm", y="Reward", palette=palette, ax=axes[0,1])
        axes[0,1].set_title("2. Standard Test (1.0x Mass)")
        
    if list_ood:
        sns.barplot(data=pd.DataFrame(list_ood), x="Algorithm", y="Reward", palette=palette, ax=axes[1,0])
        axes[1,0].set_title("3. OOD Test (3.0x Mass)")
        
    if list_adaptation:
        sns.lineplot(data=pd.DataFrame(list_adaptation), x="Step", y="Tracking Error", hue="Algorithm", palette=palette, ax=axes[1,1])
        axes[1,1].axvline(x=CFG.mass_change_step, color='r', linestyle='--', label='Mass Change')
        axes[1,1].set_title("4. Adaptation Profile")

    plt.tight_layout()
    save_path = os.path.join(CFG.result_dir, "Results.png")
    plt.savefig(save_path, dpi=300)
    print(f"Grafik kaydedildi: {save_path}")
    plt.show()

# %% [7] MAIN EXECUTION
if __name__ == "__main__":
    
    # --- KULLANIM ---
    # TRAIN_MODE = True  -> Eğitir ve kaydeder.
    # TRAIN_MODE = False -> Kayıtlı modelleri yükler ve grafikleri çizer.
    
    TRAIN_MODE = False 
    
    if TRAIN_MODE:
        train_manager("vanilla")
        train_manager("dr")
        train_manager("lstm")
    else:
        plot_comprehensive_results()