# %% [1] SETUP & LIBRARIES
import os
import glob
import random
import dataclasses
import numpy as np
import pandas as pd
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    raise ImportError("Lütfen 'sb3-contrib' kurun: pip install sb3-contrib")

DEVICE = "cpu" # Test için CPU yeterli ve stabil
print(f"[SYSTEM] Running on: {DEVICE}")

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# %% [2] CONFIGURATION (SLOW & LONG TRAJECTORY)
@dataclasses.dataclass
class ExperimentConfig:
    experiment_name: str = "MetaReacher_SlowTracking_6x"
    env_id: str = "Reacher-v4"
    seed: int = 42
    
    # Eğitim Ayarları (Mevcut modelleri kullanacağımız için burası sadece referans)
    total_timesteps: int = 5_000_000 
    n_envs: int = 8 
    
    # --- YENİ ZAMANLAMA AYARLARI ---
    max_episode_steps: int = 800     # 800 Adım (Daha uzun izleme)
    mass_change_step: int = 400      # Tam ortada değişim
    mass_multiplier: float = 6.0     # 6x Kütle
    
    # Yörünge Hızı (YAVAŞLATILDI)
    trajectory_speed: float = 0.015  # Daha yavaş, pürüzsüz daire
    trajectory_radius: float = 0.20  # Dairenin genişliği
    
    result_dir: str = "./results_5M" # Modellerin olduğu yer

CFG = ExperimentConfig()
seed_everything(CFG.seed)

# %% [3] TRAJECTORY WRAPPER
class TrajectoryReacherWrapper(gym.Wrapper):
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
        
        if hasattr(self.unwrapped, "model"):
            self.original_masses = self.unwrapped.model.body_mass.copy()
        else:
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
        
        if self.dynamic_test_mode:
            self._update_target_position(t=0)

        return self._augment_obs(obs), info

    def step(self, action):
        self.step_count += 1
        
        if self.dynamic_test_mode:
            # 1. Kütle Değişimi (Step 400)
            if self.step_count == CFG.mass_change_step:
                self._set_mass_scale(CFG.mass_multiplier)
            
            # 2. Yavaş Hedef Hareketi
            self._update_target_position(self.step_count)

        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        self.prev_action = action
        self.prev_reward = float(reward)
        self.prev_done = float(done)
        
        aug_obs = self._augment_obs(obs)
        return aug_obs, reward, terminated, truncated, info

    def _augment_obs(self, obs):
        return np.concatenate([
            obs, self.prev_action, [self.prev_reward], [self.prev_done]
        ]).astype(np.float32)

    def _set_mass_scale(self, scale: float):
        if self.original_masses is not None:
            new_mass = self.original_masses.copy()
            try: new_mass[2] *= scale 
            except: new_mass[-1] *= scale
            self.unwrapped.model.body_mass[:] = new_mass

    def _update_target_position(self, t):
        angle = t * CFG.trajectory_speed
        x = CFG.trajectory_radius * np.sin(angle)
        y = CFG.trajectory_radius * np.cos(angle)
        
        qpos = self.unwrapped.data.qpos.flat[:]
        qvel = self.unwrapped.data.qvel.flat[:]
        qpos[-2:] = np.array([x, y])
        self.unwrapped.set_state(qpos, qvel)

# %% [4] FACTORY
def make_env(rank: int, is_training=True, dr_mode=False, dynamic_test_mode=False, log_dir=None):
    def _init():
        # Uzatılmış epizot süresi
        env = gym.make(CFG.env_id, max_episode_steps=CFG.max_episode_steps)
        env = TrajectoryReacherWrapper(env, is_training, dr_mode, dynamic_test_mode)
        if log_dir: env = Monitor(env, os.path.join(log_dir, str(rank)))
        return env
    return _init

# %% [5] VISUALIZATION (Fixed Seaborn Error)
def plot_slow_trajectory_results():
    print(f"\n>>> Yavaş Yörünge Analizi ({CFG.max_episode_steps} Steps, Speed {CFG.trajectory_speed})...")
    
    plt.rcParams.update({
        "font.family": "serif", 
        "font.size": 12, 
        "lines.linewidth": 2.0, 
        "figure.autolayout": True
    })
    
    styles = {
        "Vanilla PPO": {"c": "#888888", "ls": ":",  "lw": 2.5},
        "DR-PPO":      {"c": "#444444", "ls": "--", "lw": 2.5},
        "RL²-LSTM":    {"c": "#000000", "ls": "-",  "lw": 3.0},
    }
    
    models_config = {
        "Vanilla PPO": {"file": "vanilla", "class": PPO},
        "DR-PPO":      {"file": "dr",      "class": PPO},
        "RL²-LSTM":    {"file": "lstm",    "class": RecurrentPPO},
    }
    
    list_tracking = []

    for name, conf in models_config.items():
        model_path = os.path.join(CFG.result_dir, f"{conf['file']}_model.zip")
        stats_path = os.path.join(CFG.result_dir, f"{conf['file']}_vecnorm.pkl")
        
        if not os.path.exists(model_path):
            print(f"[UYARI] Model yok: {name}"); continue
        
        print(f"[TEST] {name}...")
        agent = conf['class'].load(model_path, device="cpu")
        
        env = DummyVecEnv([make_env(rank=0, is_training=False, dynamic_test_mode=True)])
        if os.path.exists(stats_path):
            env = VecNormalize.load(stats_path, env)
            env.training = False; env.norm_reward = False
        
        # 30 Tekrar
        for ep in range(30): 
            env.seed(2000 + ep) # Yeni bir seed seti
            obs = env.reset()
            lstm_states = None
            d = np.ones((1,), dtype=bool)
            
            for t in range(CFG.max_episode_steps):
                if conf['class'] == RecurrentPPO:
                    action, lstm_states = agent.predict(obs, state=lstm_states, episode_start=d, deterministic=True)
                else:
                    action, _ = agent.predict(obs, deterministic=True)
                
                obs, reward, done, info = env.step(action)
                
                try: dist = info[0]['reward_dist']
                except: dist = abs(reward[0])
                
                list_tracking.append({
                    "Algorithm": name, "Step": t, "Tracking Error (m)": dist, "Episode": ep
                })
                d = done
                if done[0]: break
        env.close()

    # ÇİZİM
    if list_tracking:
        plt.figure(figsize=(12, 6))
        df_track = pd.DataFrame(list_tracking)
        
        sns.lineplot(
            data=df_track, x="Step", y="Tracking Error (m)", hue="Algorithm", style="Algorithm",
            palette={k:v["c"] for k,v in styles.items()}, 
            dashes=False, # Hata önleyici
            errorbar=('ci', 95)
        )
        
        # Çizgi stillerini zorla uygula
        ax = plt.gca()
        for line in ax.get_lines():
            lbl = line.get_label()
            if lbl in styles:
                line.set_linestyle(styles[lbl]["ls"])
                line.set_linewidth(styles[lbl]["lw"])
                line.set_color(styles[lbl]["c"])
                line.set_alpha(1.0)

        # Olay: KÜTLE DEĞİŞİMİ
        plt.axvline(x=CFG.mass_change_step, color='red', linestyle='-', linewidth=1.5, alpha=0.8)
        
        y_lim = plt.ylim()[1]
        plt.text(CFG.mass_change_step - 15, y_lim*0.9, "Task 1: Normal Mass", ha='right', fontsize=12, style='italic')
        plt.text(CFG.mass_change_step + 15, y_lim*0.9, "Task 2: Heavy Mass (6x)", ha='left', fontsize=12, style='italic', fontweight='bold')
        plt.text(CFG.mass_change_step, y_lim*0.95, "DYNAMICS CHANGE", ha='center', color='red', fontsize=10, backgroundcolor='white')

        plt.title(f"Trajectory Tracking Adaptation (Speed={CFG.trajectory_speed}, Mass={CFG.mass_multiplier}x)")
        plt.xlabel(f"Time Steps (0-{CFG.max_episode_steps})")
        plt.ylabel("Tracking Error (Euclidean Distance)")
        plt.legend(loc='upper left')
        plt.grid(True, ls="--", alpha=0.3)
        
        save_path = os.path.join(CFG.result_dir, "Slow_Trajectory_6x.png")
        plt.savefig(save_path, dpi=300)
        print(f"Grafik kaydedildi: {save_path}")
        plt.show()
    else:
        print("[HATA] Veri yok.")

if __name__ == "__main__":
    plot_slow_trajectory_results()