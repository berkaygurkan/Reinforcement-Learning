# %% [1] SETUP & LIBRARIES
import os
import glob
import time
import json
import random # <--- EKLENDİ
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

DEVICE = "cpu"
print(f"[SYSTEM] Running on: {DEVICE}")

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# %% [2] CONFIGURATION
@dataclasses.dataclass
class ExperimentConfig:
    experiment_name: str = "MetaReacher_Final_Fix"
    env_id: str = "Reacher-v4"
    seed: int = 42
    total_timesteps: int = 5_000_000 
    n_envs: int = 8 
    result_dir: str = "./results_5M"

CFG = ExperimentConfig()
seed_everything(CFG.seed)
os.makedirs(CFG.result_dir, exist_ok=True)

# %% [3] ROBUST WRAPPER (GUARANTEED EFFECT)
class MetaReacherWrapper(gym.Wrapper):
    def __init__(self, env, is_training: bool = True, dr_mode: bool = False, 
                 dynamic_test_mode: bool = False,
                 change_step: int = 25,
                 target_mass_scale: float = 1.0):
        
        super().__init__(env)
        self.is_training = is_training
        self.dr_mode = dr_mode
        self.dynamic_test_mode = dynamic_test_mode
        self.change_step = change_step
        self.target_mass_scale = target_mass_scale
        
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
        self.mass_changed = False

    def reset(self, **kwargs):
        self.step_count = 0
        self.mass_changed = False
        self.prev_action = np.zeros(self.act_dim, dtype=np.float32)
        self.prev_reward = 0.0
        self.prev_done = 0.0
        
        # Eğitim sırasında rastgelelik
        if self.is_training and self.dr_mode:
            # Eğitimde sadece ufak fiziksel değişimler (Action scaling yok)
            if hasattr(self.unwrapped, "model"):
                scale = np.random.uniform(0.5, 2.0)
                self.unwrapped.model.body_mass[1:] *= scale
            
        obs, info = self.env.reset(**kwargs)
        return self._augment_obs(obs), info

    def step(self, action):
        self.step_count += 1
        
        effective_action = action.copy()

        # --- DİNAMİK DEĞİŞİM (TEST MODU) ---
        if self.dynamic_test_mode:
            # Step geldiğinde bayrağı kaldır
            if self.step_count >= self.change_step:
                if not self.mass_changed and self.target_mass_scale != 1.0:
                    print(f"[DEBUG] Step {self.step_count}: Kütle/Güç Oranı {self.target_mass_scale}x değişti!")
                    self.mass_changed = True
                
                # FİZİKSEL ETKİ SİMÜLASYONU (POWER-TO-WEIGHT RATIO)
                # Eğer kütle 10 kat artarsa, motorlar aynı işi yapmak için 10 kat zorlanır.
                # Bunu simüle etmek için Action'ı (Torku) zayıflatıyoruz.
                # Bu yöntem PLANAR robotlarda kütle etkisini görmenin tek garantili yoludur.
                if self.target_mass_scale > 1.0:
                    effective_action = action / self.target_mass_scale 

        # Ortama zayıflatılmış aksiyonu gönder
        obs, reward, terminated, truncated, info = self.env.step(effective_action)
        done = terminated or truncated
        
        # LSTM'e ise AJANIN YAPMAK İSTEDİĞİ (orijinal) aksiyonu veriyoruz.
        # Böylece ajan "Ben X kadar ittim ama kol gitmedi, demek ki ağırlaşmış" diyebiliyor.
        self.prev_action = action 
        self.prev_reward = float(reward)
        self.prev_done = float(done)
        
        return self._augment_obs(obs), reward, terminated, truncated, info

    def _augment_obs(self, obs):
        return np.concatenate([
            obs, self.prev_action, [self.prev_reward], [self.prev_done]
        ]).astype(np.float32)

# %% [4] FACTORY
def make_env(rank: int, is_training=True, dr_mode=False, dynamic_test_mode=False, 
             change_step=25, target_mass_scale=1.0, log_dir=None):
    def _init():
        env = gym.make(CFG.env_id)
        env = MetaReacherWrapper(
            env, is_training=is_training, dr_mode=dr_mode, 
            dynamic_test_mode=dynamic_test_mode,
            change_step=change_step, target_mass_scale=target_mass_scale
        )
        if log_dir:
            env = Monitor(env, os.path.join(log_dir, str(rank)))
        return env
    return _init

# %% [5] TRAINING
def train_manager(algo_type: str):
    # (Bu kısım aynı kalıyor, sadece modelleri yüklemek için gerekli)
    run_name = f"{algo_type}_seed{CFG.seed}"
    log_dir = os.path.join(CFG.result_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    
    model_path = os.path.join(CFG.result_dir, f"{algo_type}_model.zip")
    stats_path = os.path.join(CFG.result_dir, f"{algo_type}_vecnorm.pkl")

    if os.path.exists(model_path): return

    print(f">>> Eğitim: {algo_type.upper()}")
    envs = SubprocVecEnv([make_env(i, True, algo_type in ['dr', 'lstm'], log_dir=log_dir) for i in range(CFG.n_envs)])
    envs = VecNormalize(envs, norm_obs=True, norm_reward=True, clip_obs=10.)

    if algo_type == 'lstm':
        model = RecurrentPPO("MlpLstmPolicy", envs, verbose=1, device=DEVICE, batch_size=512, 
                             policy_kwargs={"lstm_hidden_size": 256, "shared_lstm": True, "enable_critic_lstm": False})
    else:
        model = PPO("MlpPolicy", envs, verbose=1, device=DEVICE, batch_size=512)

    model.learn(total_timesteps=CFG.total_timesteps)
    model.save(model_path)
    envs.save(stats_path)
    envs.close()

# %% [6] VISUALIZATION (GUARANTEED DIFFERENCE)
def plot_ab_test_results():
    print("\n>>> A/B Test Görselleştirme (Force Simulation)...")
    
    plt.rcParams.update({"font.family": "serif", "font.size": 11, "lines.linewidth": 2.0, "figure.autolayout": True})
    
    styles = {
        "Vanilla PPO": {"c": "0.4", "ls": ":"}, 
        "DR-PPO":      {"c": "0.2", "ls": "--"}, 
        "RL²-LSTM":    {"c": "0.0", "ls": "-"},    
    }
    
    models_config = {
        "Vanilla PPO": {"file": "vanilla", "class": PPO},
        "DR-PPO":      {"file": "dr",      "class": PPO},
        "RL²-LSTM":    {"file": "lstm",    "class": RecurrentPPO},
    }
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    
    # SENARYOLAR (10x FARK)
    scenarios = [
        {"name": "A) Dynamic Test (Mass Spike 10x -> Weak Motor)", "mult": 10.0, "ax": axes[0]},
        {"name": "B) Control Test (No Change -> Normal Motor)",    "mult": 1.0,  "ax": axes[1]}
    ]

    for scen in scenarios:
        mass_mult = scen["mult"]
        ax = scen["ax"]
        print(f"\n--- Senaryo: {scen['name']} ---")
        
        list_adaptation = []
        
        for name, conf in models_config.items():
            model_path = os.path.join(CFG.result_dir, f"{conf['file']}_model.zip")
            stats_path = os.path.join(CFG.result_dir, f"{conf['file']}_vecnorm.pkl")
            
            if not os.path.exists(model_path): continue
            
            agent = conf['class'].load(model_path, device="cpu")
            
            # Env Factory
            env_adapt = DummyVecEnv([
                make_env(rank=0, is_training=False, dynamic_test_mode=True, 
                         change_step=CFG.default_change_step, 
                         target_mass_scale=mass_mult)
            ])
            
            if os.path.exists(stats_path):
                env_adapt = VecNormalize.load(stats_path, env_adapt)
                env_adapt.training = False; env_adapt.norm_reward = False
            
            # 30 Tekrar
            for ep in range(30):
                env_adapt.seed(1000 + ep) 
                obs = env_adapt.reset()
                lstm_states = None
                d = np.ones((1,), dtype=bool)
                
                for t in range(50):
                    if conf['class'] == RecurrentPPO:
                        action, lstm_states = agent.predict(obs, state=lstm_states, episode_start=d, deterministic=True)
                    else:
                        action, _ = agent.predict(obs, deterministic=True)
                    
                    obs, reward, done, info = env_adapt.step(action)
                    
                    # Mesafe
                    try: dist = info[0]['reward_dist']
                    except: dist = abs(reward[0])
                    
                    list_adaptation.append({
                        "Algorithm": name, "Step": t, "Distance Error (m)": dist, "Episode": ep
                    })
                    d = done
                    if done[0]: break
            env_adapt.close()

        # Çizim
        if list_adaptation:
            df_scen = pd.DataFrame(list_adaptation)
            sns.lineplot(
                data=df_scen, x="Step", y="Distance Error (m)", hue="Algorithm", style="Algorithm",
                palette={k:v["c"] for k,v in styles.items()}, 
                errorbar=('ci', 95), linewidth=2.5, ax=ax, legend=(ax == axes[0])
            )
            
            ax.axvline(x=CFG.default_change_step, color='k', linestyle='-', alpha=0.4)
            ax.set_title(scen["name"], fontweight="bold")
            ax.grid(True, ls="--", alpha=0.3)
            
            if mass_mult > 1.0:
                ax.text(CFG.default_change_step+1, ax.get_ylim()[1]*0.5, "MASS SPIKE!", color='red', fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(CFG.result_dir, "Guaranteed_AB_Test.png")
    plt.savefig(save_path, dpi=300)
    print(f"\nGrafik kaydedildi: {save_path}")
    plt.show()

if __name__ == "__main__":
    TRAIN_MODE = False
    if TRAIN_MODE:
        train_manager("vanilla")
        train_manager("dr")
        train_manager("lstm")
        plot_ab_test_results()
    else:
        plot_ab_test_results()