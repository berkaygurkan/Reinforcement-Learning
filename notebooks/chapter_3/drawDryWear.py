# %% [1] KURULUM VE AYARLAR
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

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# RecurrentPPO Kontrolü
try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    raise ImportError("Lütfen 'sb3-contrib' kurun: pip install sb3-contrib")

# Grafik Ayarları (Akademik Stil)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "axes.labelsize": 12,
    "font.size": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 2.0,
    "figure.autolayout": True
})

# Renk ve Stil Şeması (Siyah-Beyaz Baskı Dostu)
STYLES = {
    "Vanilla PPO": {"c": "0.4", "ls": ":",  "m": "o", "h": "///"}, # Açık Gri
    "DR-PPO":      {"c": "0.2", "ls": "--", "m": "s", "h": "..."}, # Koyu Gri
    "RL²-LSTM":    {"c": "0.0", "ls": "-",  "m": "^", "h": ""},    # Siyah
}

MODELS_CONFIG = {
    "Vanilla PPO": {"file": "vanilla", "class": PPO},
    "DR-PPO":      {"file": "dr",      "class": PPO},
    "RL²-LSTM":    {"file": "lstm",    "class": RecurrentPPO},
}

@dataclasses.dataclass
class PlotConfig:
    env_id: str = "Reacher-v4"
    seed: int = 42
    result_dir: str = "./results_friction_training" # Yeni eğitim klasörü
    
    # Test Parametreleri
    n_test_episodes: int = 50   # OOD testi için tekrar sayısı
    n_adapt_episodes: int = 30  # Adaptasyon testi için tekrar sayısı
    
    # Senaryo Parametreleri
    change_step: int = 5       # A/B testinde değişim anı
    ood_friction: float = 4.0   # OOD testindeki zorluk
    ab_friction: float = 20.0    # A/B testindeki ani şok miktarı

CFG = PlotConfig()

# %% [2] TEST WRAPPER (FRICTION FOCUSED)
class FrictionTestWrapper(gym.Wrapper):
    def __init__(self, env, 
                 dynamic_mode: bool = False,    # True ise adım içinde değişim olur (A/B)
                 initial_friction: float = 0.0, # Başlangıç sürtünmesi
                 target_friction: float = 0.0,  # Hedef sürtünme (Dynamic mod için)
                 change_step: int = 25):        # Değişim adımı
        super().__init__(env)
        self.dynamic_mode = dynamic_mode
        self.initial_friction = initial_friction
        self.target_friction = target_friction
        self.change_step = change_step
        
        self.act_dim = env.action_space.shape[0]
        self.obs_dim = env.observation_space.shape[0]
        # LSTM için augmented obs
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim + self.act_dim + 2,), dtype=np.float32
        )
        
        self.prev_action = np.zeros(self.act_dim, dtype=np.float32)
        self.prev_reward = 0.0
        self.prev_done = 0.0
        self.step_count = 0
        self.changed = False

    def reset(self, **kwargs):
        self.step_count = 0
        self.changed = False
        self.prev_action = np.zeros(self.act_dim, dtype=np.float32)
        self.prev_reward = 0.0
        self.prev_done = 0.0
        
        # Başlangıç sürtünmesini ayarla
        if hasattr(self.unwrapped, "model"):
            self.unwrapped.model.dof_frictionloss[:] = self.initial_friction
            
        obs, info = self.env.reset(**kwargs)
        return self._augment_obs(obs), info

    def step(self, action):
        self.step_count += 1
        
        # --- DİNAMİK DEĞİŞİM (A/B Testi İçin) ---
        if self.dynamic_mode and not self.changed and self.step_count >= self.change_step:
            if hasattr(self.unwrapped, "model"):
                self.unwrapped.model.dof_frictionloss[:] = self.target_friction
            self.changed = True

        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        self.prev_action = action
        self.prev_reward = float(reward)
        self.prev_done = float(done)
        
        return self._augment_obs(obs), reward, terminated, truncated, info

    def _augment_obs(self, obs):
        return np.concatenate([
            obs, self.prev_action, [self.prev_reward], [self.prev_done]
        ]).astype(np.float32)

# Helper Factory
def make_test_env(dynamic_mode=False, initial_friction=0.0, target_friction=0.0, change_step=25):
    def _init():
        env = gym.make(CFG.env_id)
        env = FrictionTestWrapper(env, dynamic_mode, initial_friction, target_friction, change_step)
        return env
    return _init

# %% [3] PLOT 1: LEARNING CURVES
def plot_learning_curves():
    print(">>> 1. Learning Curves Çiziliyor...")
    df_list = []
    
    for name, conf in MODELS_CONFIG.items():
        log_path = os.path.join(CFG.result_dir, f"{conf['file']}_seed{CFG.seed}", "*.monitor.csv")
        files = glob.glob(log_path)
        
        for f in files:
            try:
                df = pd.read_csv(f, skiprows=1)
                if 'r' in df.columns:
                    # Veriyi seyrelterek yumuşat (Smoothing)
                    df = df.iloc[::50, :] 
                    df['Algorithm'] = name
                    df['Timesteps'] = df['l'].cumsum()
                    df['Reward'] = df['r'].ewm(alpha=0.05).mean()
                    df_list.append(df)
            except: pass
            
    if not df_list:
        print("[UYARI] Log dosyası bulunamadı. Eğitim yapıldı mı?")
        return

    df_all = pd.concat(df_list)
    
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df_all, x="Timesteps", y="Reward", hue="Algorithm", 
                 palette={k:v["c"] for k,v in STYLES.items()}, style="Algorithm")
    
    plt.title("Training Performance (Adaptive Friction Task)")
    plt.xlabel("Timesteps")
    plt.ylabel("Average Reward")
    plt.grid(True, ls="--", alpha=0.3)
    
    save_path = os.path.join(CFG.result_dir, "Fig1_Learning_Curve.png")
    plt.savefig(save_path, dpi=300)
    print(f"[KAYDEDİLDİ] {save_path}")
    plt.show()

# %% [4] PLOT 2: OOD ROBUSTNESS (BAR PLOT)
def plot_ood_robustness():
    print(">>> 2. OOD Robustness Testi Yapılıyor...")
    results = []
    
    # Test Durumları: Normal (0.0) vs Extreme (CFG.ood_friction)
    conditions = [
        ("Standard (0.0 Friction)", 0.0),
        (f"OOD ({CFG.ood_friction} Friction)", CFG.ood_friction)
    ]
    
    for name, conf in MODELS_CONFIG.items():
        path = os.path.join(CFG.result_dir, f"{conf['file']}_model.zip")
        stats = os.path.join(CFG.result_dir, f"{conf['file']}_vecnorm.pkl")
        if not os.path.exists(path): continue
        
        agent = conf['class'].load(path, device="cpu")
        
        for label, fric_val in conditions:
            # Statik Env (Dynamic False)
            env = DummyVecEnv([make_test_env(dynamic_mode=False, initial_friction=fric_val)])
            if os.path.exists(stats): env = VecNormalize.load(stats, env); env.training = False
            
            for _ in range(CFG.n_test_episodes):
                obs = env.reset()
                done = False
                ep_rew = 0
                lstm_states = None
                d = np.ones((1,), dtype=bool)
                while not done:
                    if conf['class'] == RecurrentPPO:
                        action, lstm_states = agent.predict(obs, state=lstm_states, episode_start=d, deterministic=True)
                    else:
                        action, _ = agent.predict(obs, deterministic=True)
                    obs, r, done_arr, _ = env.step(action)
                    ep_rew += r[0]
                    done = done_arr[0]
                    d = done_arr
                results.append({"Algorithm": name, "Condition": label, "Reward": ep_rew})
            env.close()

    if not results: return

    plt.figure(figsize=(8, 6))
    df_res = pd.DataFrame(results)
    ax = sns.barplot(data=df_res, x="Condition", y="Reward", hue="Algorithm",
                     palette={k:v["c"] for k,v in STYLES.items()},
                     edgecolor="black", errorbar="sd", capsize=0.1)
    
    # Hatch (Desen) Ekleme
    for i, container in enumerate(ax.containers):
        # Seaborn bar sırası ile STYLES anahtarlarını eşleştir
        # Bu kısım bazen sıraya göre değişebilir, garanti yöntem algo ismine bakmaktır ama basitlik için:
        algo_name = list(STYLES.keys())[i % 3] 
        pattern = STYLES[algo_name]["h"]
        for bar in container: bar.set_hatch(pattern)

    plt.title("Robustness Analysis (Standard vs High Friction)")
    plt.grid(axis='y', ls="--", alpha=0.3)
    
    save_path = os.path.join(CFG.result_dir, "Fig2_Robustness_Bar.png")
    plt.savefig(save_path, dpi=300)
    print(f"[KAYDEDİLDİ] {save_path}")
    plt.show()

# %% [5] PLOT 3: A/B ADAPTATION TEST (SUBPLOTS)
def plot_ab_adaptation():
    print(">>> 3. A/B Adaptasyon Testi Yapılıyor...")
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True , sharey=True)
    
    # Senaryolar: A (Spike) vs B (Kontrol)
    # A: 0.0 -> CFG.ab_friction (Step 25)
    # B: 0.0 -> 0.0 (Step 25 - Değişim Yok)
    scenarios = [
        {"name": f"A) Sudden Failure (Friction 0.0 -> {CFG.ab_friction})", 
         "init": 0.0, "target": CFG.ab_friction, "ax": axes[0]},
        {"name": "B) Control (No Friction Change)", 
         "init": 0.0, "target": 0.0, "ax": axes[1]}
    ]
    
    OBS_STEPS = 60 # 25 öncesi + sonrası
    
    for scen in scenarios:
        ax = scen["ax"]
        results = []
        
        for name, conf in MODELS_CONFIG.items():
            path = os.path.join(CFG.result_dir, f"{conf['file']}_model.zip")
            stats = os.path.join(CFG.result_dir, f"{conf['file']}_vecnorm.pkl")
            if not os.path.exists(path): continue
            
            agent = conf['class'].load(path, device="cpu")
            
            # Env Setup
            env = DummyVecEnv([make_test_env(
                dynamic_mode=True, 
                initial_friction=scen["init"], 
                target_friction=scen["target"],
                change_step=CFG.change_step
            )])
            if os.path.exists(stats): env = VecNormalize.load(stats, env); env.training = False
            
            for ep in range(CFG.n_adapt_episodes):
                env.seed(1000 + ep) # Sabit seed = Adil karşılaştırma
                obs = env.reset()
                lstm_states = None
                d = np.ones((1,), dtype=bool)
                
                for t in range(OBS_STEPS):
                    if conf['class'] == RecurrentPPO:
                        action, lstm_states = agent.predict(obs, state=lstm_states, episode_start=d, deterministic=True)
                    else:
                        action, _ = agent.predict(obs, deterministic=True)
                    
                    obs, reward, done, info = env.step(action)
                    
                    # Mesafe Hatası (Mutlak Değer)
                    try: dist = info[0]['reward_dist']
                    except: dist = abs(reward[0])
                    
                    results.append({"Algorithm": name, "Step": t, "Error": dist, "Episode": ep})
                    d = done
                    if done[0]: break
            env.close()
            
        if results:
            df_res = pd.DataFrame(results)
            sns.lineplot(data=df_res, x="Step", y="Error", hue="Algorithm", style="Algorithm",
                         palette={k:v["c"] for k,v in STYLES.items()}, 
                         errorbar=('ci', 95), linewidth=2.5, ax=ax, legend=(ax == axes[0]))
            
            # Olay Çizgisi
            ax.axvline(x=CFG.change_step, color='red', ls='-', alpha=0.6)
            ax.set_title(scen["name"], fontweight="bold")
            ax.grid(True, ls="--", alpha=0.3)
            ax.set_ylabel("Distance Error (m)")
            
            if scen["target"] > scen["init"]:
                ax.text(CFG.change_step + 1, ax.get_ylim()[1]*0.9, "FAILURE ONSET", color='red', fontweight='bold')

    axes[1].set_xlabel("Time Steps")
    plt.tight_layout()
    
    save_path = os.path.join(CFG.result_dir, "Fig3_AB_Adaptation.png")
    plt.savefig(save_path, dpi=300)
    print(f"[KAYDEDİLDİ] {save_path}")
    plt.show()

# %% [6] MAIN EXECUTION
if __name__ == "__main__":
    if not os.path.exists(CFG.result_dir):
        print(f"[HATA] Klasör bulunamadı: {CFG.result_dir}")
        print("Lütfen önce eğitim kodunu çalıştırın.")
    else:
        plot_learning_curves()
        plot_ood_robustness()
        plot_ab_adaptation()
        print("\n>>> Tüm grafikler başarıyla oluşturuldu.")