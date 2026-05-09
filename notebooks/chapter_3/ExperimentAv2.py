# experiment_kusursuz_distance.py
import os
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
from stable_baselines3.common.logger import configure

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    raise ImportError("sb3-contrib kurulu değil: pip install sb3-contrib")


@dataclasses.dataclass
class ExperimentConfig:
    experiment_name: str = "MetaReacher_Thesis_DistanceBased"
    env_id: str = "Reacher-v5"
    seed: int = 42

    total_timesteps: int = 1_000_000
    n_envs: int = 8

    dr_range: tuple = (0.5, 2.0)
    meta_change_prob: float = 0.4
    meta_change_step_range: tuple = (15, 35)

    mass_change_step: int = 25
    mass_multiplier_test: float = 3.0

    eval_episodes_std_ood: int = 20
    eval_episodes_adapt: int = 10
    eval_max_steps_adapt: int = 100

    # >>> KRİTİK: Burayı, modellerin gerçekten durduğu klasöre ayarla <<<
    # Örn: "./results_reacher_kusursuz" veya "./results_reacher_kusursuz_fixedim"
    result_dir: str = "./results_reacher_kusursuz"


CFG = ExperimentConfig()

random.seed(CFG.seed)
np.random.seed(CFG.seed)
torch.manual_seed(CFG.seed)
os.makedirs(CFG.result_dir, exist_ok=True)

DEVICE = "cpu"
print(f"[SYSTEM] Running on: {DEVICE}")
print(f"[SYSTEM] Result dir: {CFG.result_dir}")


class SimpleWrapper(gym.Wrapper):
    def __init__(self, env, mode: str = "vanilla", fixed_mass: float = None):
        super().__init__(env)
        self.mode = mode
        self.fixed_mass = fixed_mass
        self.step_count = 0
        self.base_obs_dim = env.observation_space.shape[0]
        if hasattr(self.unwrapped, "model"):
            self.original_masses = self.unwrapped.model.body_mass.copy()
        else:
            self.original_masses = None

    def reset(self, **kwargs):
        self.step_count = 0
        scale = self.fixed_mass if self.fixed_mass is not None else (
            np.random.uniform(*CFG.dr_range) if self.mode == "dr" else 1.0
        )
        self._set_mass_scale(scale)
        return self.env.reset(**kwargs)

    def step(self, action):
        self.step_count += 1
        return self.env.step(action)

    def _set_mass_scale(self, scale: float):
        if self.original_masses is not None:
            new_mass = self.original_masses.copy()
            new_mass[2:] *= float(scale)
            self.unwrapped.model.body_mass[:] = new_mass


class MetaReacherWrapper(gym.Wrapper):
    def __init__(self, env, fixed_mass: float = None):
        super().__init__(env)
        self.fixed_mass = fixed_mass
        self.step_count = 0

        self.act_dim = env.action_space.shape[0]
        self.obs_dim = env.observation_space.shape[0]
        self.base_obs_dim = self.obs_dim

        self.augmented_dim = self.obs_dim + self.act_dim + 2
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.augmented_dim,), dtype=np.float32
        )

        self.prev_action = np.zeros(self.act_dim, dtype=np.float32)
        self.prev_reward = 0.0
        self.prev_done = 0.0

        if hasattr(self.unwrapped, "model"):
            self.original_masses = self.unwrapped.model.body_mass.copy()
        else:
            self.original_masses = None

    def reset(self, **kwargs):
        self.step_count = 0
        self.prev_action = np.zeros(self.act_dim, dtype=np.float32)
        self.prev_reward = 0.0
        self.prev_done = 0.0

        scale = self.fixed_mass if self.fixed_mass is not None else 1.0
        self._set_mass_scale(scale)

        obs, info = self.env.reset(**kwargs)
        return self._augment_obs(obs), info

    def step(self, action):
        self.step_count += 1

        if self.fixed_mass is None:
            if np.random.rand() < CFG.meta_change_prob:
                if CFG.meta_change_step_range[0] <= self.step_count <= CFG.meta_change_step_range[1]:
                    scale = np.random.uniform(2.0, 4.0)
                    self._set_mass_scale(scale)

        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        aug_obs = self._augment_obs(obs)
        self.prev_action = action.copy()
        self.prev_reward = float(reward)
        self.prev_done = float(done)

        return aug_obs, reward, terminated, truncated, info

    def _augment_obs(self, obs):
        return np.concatenate([obs, self.prev_action, [self.prev_reward], [self.prev_done]]).astype(np.float32)

    def _set_mass_scale(self, scale: float):
        if self.original_masses is not None:
            new_mass = self.original_masses.copy()
            new_mass[2:] *= float(scale)
            self.unwrapped.model.body_mass[:] = new_mass


def make_env(algo_type: str, rank: int, log_dir=None):
    def _init():
        env = gym.make(CFG.env_id)
        if algo_type == "lstm":
            env = MetaReacherWrapper(env, fixed_mass=None)
        else:
            mode = "dr" if algo_type == "dr" else "vanilla"
            env = SimpleWrapper(env, mode=mode, fixed_mass=None)
        if log_dir:
            env = Monitor(env, os.path.join(log_dir, str(rank)))
        return env
    return _init


def train_manager(algo_type: str):
    run_name = f"{algo_type}_seed{CFG.seed}"
    log_dir = os.path.join(CFG.result_dir, "tb_logs", run_name)
    model_path = os.path.join(CFG.result_dir, f"{algo_type}_model.zip")
    stats_path = os.path.join(CFG.result_dir, f"{algo_type}_vecnorm.pkl")

    if os.path.exists(model_path) and os.path.exists(stats_path):
        print(f"[SKIP] {algo_type.upper()} zaten var.")
        return

    print(f"\n>>> {algo_type.upper()} eğitimi başlıyor...")

    envs = SubprocVecEnv([
        make_env(algo_type=algo_type, rank=i, log_dir=os.path.join(CFG.result_dir, "monitor", run_name))
        for i in range(CFG.n_envs)
    ])
    envs = VecNormalize(envs, norm_obs=True, norm_reward=True, clip_obs=10.0)
    new_logger = configure(log_dir, ["tensorboard"])

    if algo_type == "lstm":
        policy_kwargs = dict(lstm_hidden_size=256, n_lstm_layers=2, shared_lstm=True, enable_critic_lstm=False)
        model = RecurrentPPO(
            "MlpLstmPolicy", envs, verbose=1, device=DEVICE,
            batch_size=512, n_steps=2048, learning_rate=3e-4, ent_coef=0.01,
            policy_kwargs=policy_kwargs, tensorboard_log=log_dir
        )
    else:
        model = PPO(
            "MlpPolicy", envs, verbose=1, device=DEVICE,
            batch_size=512, n_steps=2048, learning_rate=3e-4, ent_coef=0.01,
            tensorboard_log=log_dir
        )

    model.set_logger(new_logger)
    model.learn(total_timesteps=CFG.total_timesteps)
    model.save(model_path)
    envs.save(stats_path)
    envs.close()

    print(f"[SAVED] {algo_type.upper()} -> {model_path}")
    print(f"[SAVED] VecNormalize -> {stats_path}")


def _get_base_env_from_vec(vec_env):
    return vec_env.venv.envs[0] if hasattr(vec_env, "venv") else vec_env.envs[0]


def _find_wrapper_with_mass_setter(env):
    e = env
    while True:
        if hasattr(e, "_set_mass_scale"):
            return e
        if hasattr(e, "env"):
            e = e.env
        else:
            return None


def _set_mass(vec_env, scale: float):
    base = _get_base_env_from_vec(vec_env)
    w = _find_wrapper_with_mass_setter(base)
    if w is None:
        raise RuntimeError("Wrapper zincirinde _set_mass_scale bulunamadı.")
    w._set_mass_scale(scale)
    if hasattr(w, "fixed_mass"):
        w.fixed_mass = scale


def _get_base_obs_dim(vec_env, fallback: int = 10):
    base = _get_base_env_from_vec(vec_env)
    if hasattr(base, "base_obs_dim"):
        return int(base.base_obs_dim)
    # wrapper zinciri
    e = base
    while hasattr(e, "env"):
        e = e.env
        if hasattr(e, "base_obs_dim"):
            return int(e.base_obs_dim)
    return int(fallback)


def _compute_reacher_distance_from_obs(raw_obs_1d: np.ndarray) -> float:
    if raw_obs_1d.shape[0] < 10:
        raise ValueError(f"Obs boyutu çok küçük: {raw_obs_1d.shape[0]} (>=10 olmalı)")
    tip = raw_obs_1d[6:8]
    target = raw_obs_1d[8:10]
    return float(np.linalg.norm(tip - target))


def plot_results():
    print("\n>>> Test ve Görselleştirme Başlıyor (Distance-based metrics)...")

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
    palette = {"Vanilla PPO": "#e74c3c", "DR-PPO": "#f1c40f", "Meta-LSTM PPO": "#2ecc71"}
    models = {"Vanilla PPO": "vanilla", "DR-PPO": "dr", "Meta-LSTM PPO": "lstm"}

    std_rows, ood_rows, adapt_rows = [], [], []

    # Erken uyarı: klasörde model var mı?
    expected = [os.path.join(CFG.result_dir, f"{k}_model.zip") for k in ["vanilla", "dr", "lstm"]]
    if not any(os.path.exists(p) for p in expected):
        print("\n[HATA] Bu result_dir içinde hiç model bulunamadı.")
        print("      CFG.result_dir yanlış klasörü gösteriyor olabilir.")
        print("      Beklenen örnek dosyalar:")
        for p in expected:
            print("       -", p)
        print("\nÇIKIŞ (KeyError vermeden).")
        return

    for disp_name, file in models.items():
        model_path = os.path.join(CFG.result_dir, f"{file}_model.zip")
        stats_path = os.path.join(CFG.result_dir, f"{file}_vecnorm.pkl")

        if not os.path.exists(model_path):
            print(f"[ATLA] {disp_name} modeli yok: {model_path}")
            continue
        if not os.path.exists(stats_path):
            print(f"[ATLA] {disp_name} vecnorm yok: {stats_path}")
            continue

        is_lstm = (file == "lstm")
        agent_class = RecurrentPPO if is_lstm else PPO
        agent = agent_class.load(model_path, device="cpu")

        def make_test_env(fixed_mass: float):
            def _init():
                env = gym.make(CFG.env_id)
                if is_lstm:
                    env = MetaReacherWrapper(env, fixed_mass=fixed_mass)
                else:
                    mode = "dr" if file == "dr" else "vanilla"
                    env = SimpleWrapper(env, mode=mode, fixed_mass=fixed_mass)
                return env

            env = DummyVecEnv([_init])
            env = VecNormalize.load(stats_path, env)
            env.training = False
            env.norm_reward = False
            return env

        # STANDARD & OOD
        for mass, bucket in [(1.0, std_rows), (CFG.mass_multiplier_test, ood_rows)]:
            env = make_test_env(fixed_mass=mass)
            base_dim = _get_base_obs_dim(env)

            ep_rewards, ep_mean_dist, ep_final_dist = [], [], []

            for _ in range(CFG.eval_episodes_std_ood):
                obs = env.reset()
                done = False
                lstm_states = None
                episode_starts = np.ones((1,), dtype=bool)

                total_rew = 0.0
                dists = []

                while not done:
                    if is_lstm:
                        action, lstm_states = agent.predict(
                            obs, state=lstm_states, episode_start=episode_starts, deterministic=True
                        )
                    else:
                        action, _ = agent.predict(obs, deterministic=True)

                    obs, rew, done_arr, _ = env.step(action)
                    total_rew += float(rew[0])
                    done = bool(done_arr[0])
                    episode_starts = done_arr

                    raw = env.unnormalize_obs(obs) if hasattr(env, "unnormalize_obs") else obs
                    base_obs = raw[0, :base_dim]
                    dists.append(_compute_reacher_distance_from_obs(base_obs))

                ep_rewards.append(total_rew)
                ep_mean_dist.append(float(np.mean(dists)))
                ep_final_dist.append(float(dists[-1]))

            bucket.append({
                "Algorithm": disp_name,
                "RewardMean": float(np.mean(ep_rewards)),
                "RewardStd": float(np.std(ep_rewards)),
                "DistMean": float(np.mean(ep_mean_dist)),
                "DistMeanStd": float(np.std(ep_mean_dist)),
                "FinalDistMean": float(np.mean(ep_final_dist)),
                "FinalDistStd": float(np.std(ep_final_dist)),
            })
            env.close()

        # ADAPTATION
        env = make_test_env(fixed_mass=1.0)
        base_dim = _get_base_obs_dim(env)

        for ep in range(CFG.eval_episodes_adapt):
            obs = env.reset()
            done = False
            lstm_states = None
            episode_starts = np.ones((1,), dtype=bool)
            step_count = 0

            while (not done) and (step_count < CFG.eval_max_steps_adapt):
                step_count += 1

                if is_lstm:
                    action, lstm_states = agent.predict(
                        obs, state=lstm_states, episode_start=episode_starts, deterministic=True
                    )
                else:
                    action, _ = agent.predict(obs, deterministic=True)

                obs, rew, done_arr, _ = env.step(action)
                done = bool(done_arr[0])
                episode_starts = done_arr

                raw = env.unnormalize_obs(obs) if hasattr(env, "unnormalize_obs") else obs
                base_obs = raw[0, :base_dim]
                dist = _compute_reacher_distance_from_obs(base_obs)

                adapt_rows.append({
                    "Algorithm": disp_name,
                    "Episode": int(ep),
                    "Step": int(step_count),
                    "DistanceError": float(dist)
                })

                if step_count == CFG.mass_change_step:
                    _set_mass(env, CFG.mass_multiplier_test)

        env.close()

    # Eğer hiçbir model eval edilmediyse temiz çık
    if len(std_rows) == 0 or len(ood_rows) == 0 or len(adapt_rows) == 0:
        print("\n[HATA] Eval edilebilen model yok veya metrikler boş kaldı.")
        print("      result_dir doğru mu? vecnorm dosyaları mevcut mu?")
        return

    df_std = pd.DataFrame(std_rows)
    df_ood = pd.DataFrame(ood_rows)
    df_adapt = pd.DataFrame(adapt_rows)

    df_std.to_csv(os.path.join(CFG.result_dir, "metrics_standard.csv"), index=False)
    df_ood.to_csv(os.path.join(CFG.result_dir, "metrics_ood.csv"), index=False)
    df_adapt.to_csv(os.path.join(CFG.result_dir, "metrics_adaptation_steps.csv"), index=False)

    df_merge = df_std.merge(df_ood, on="Algorithm", suffixes=("_Std", "_OOD"))
    df_merge["DeltaDistMean_OOD_minus_Std"] = df_merge["DistMean_OOD"] - df_merge["DistMean_Std"]
    df_merge["DeltaFinalDist_OOD_minus_Std"] = df_merge["FinalDistMean_OOD"] - df_merge["FinalDistMean_Std"]

    print("\n[THESIS METRICS] Distance Degradation (OOD-Std) (Lower is better):")
    print(df_merge[[
        "Algorithm",
        "DistMean_Std","DistMean_OOD","DeltaDistMean_OOD_minus_Std",
        "FinalDistMean_Std","FinalDistMean_OOD","DeltaFinalDist_OOD_minus_Std"
    ]].to_string(index=False))

    pre = df_adapt[df_adapt["Step"] <= CFG.mass_change_step].groupby("Algorithm")["DistanceError"].sum().rename("AUC_Pre")
    post = df_adapt[df_adapt["Step"] > CFG.mass_change_step].groupby("Algorithm")["DistanceError"].sum().rename("AUC_Post")
    auc = pd.concat([pre, post], axis=1).fillna(0.0)
    auc["AUC_Total"] = auc["AUC_Pre"] + auc["AUC_Post"]
    auc = auc.reset_index()

    print("\n[THESIS METRICS] Adaptation AUC (Distance, Lower is better):")
    print(auc.to_string(index=False))

    df_merge.to_csv(os.path.join(CFG.result_dir, "thesis_distance_degradation.csv"), index=False)
    auc.to_csv(os.path.join(CFG.result_dir, "thesis_adaptation_auc.csv"), index=False)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    sns.barplot(
        data=df_std, x="Algorithm", y="DistMean",
        hue="Algorithm", palette=palette, legend=False, ax=axes[0, 0]
    )
    axes[0, 0].set_title("Standard (1.0x Mass) - Mean Distance (Lower better)")
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("Mean Distance")

    sns.barplot(
        data=df_ood, x="Algorithm", y="DistMean",
        hue="Algorithm", palette=palette, legend=False, ax=axes[0, 1]
    )
    axes[0, 1].set_title(f"OOD ({CFG.mass_multiplier_test}x Mass) - Mean Distance (Lower better)")
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("Mean Distance")

    sns.lineplot(
        data=df_adapt, x="Step", y="DistanceError",
        hue="Algorithm", palette=palette, ax=axes[1, 0]
    )
    axes[1, 0].axvline(x=CFG.mass_change_step, linestyle="--", color="red", label="Mass Change")
    axes[1, 0].set_title("Adaptation (Distance to Goal, Lower better)")
    axes[1, 0].set_ylabel("DistanceError")
    axes[1, 0].legend()

    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.5, 0.5,
        f"Tensorboard:\n"
        f"tensorboard --logdir {CFG.result_dir}/tb_logs\n\n"
        f"CSV outputs:\n"
        f"- metrics_standard.csv\n"
        f"- metrics_ood.csv\n"
        f"- thesis_distance_degradation.csv\n"
        f"- thesis_adaptation_auc.csv",
        ha="center", va="center", fontsize=12, transform=axes[1, 1].transAxes
    )

    plt.tight_layout()
    out_path = os.path.join(CFG.result_dir, "Final_Results_DistanceBased.png")
    plt.savefig(out_path, dpi=300)
    plt.show()

    print(f"\nGrafik kaydedildi: {out_path}")
    print(f"Tensorboard için: tensorboard --logdir {os.path.join(CFG.result_dir, 'tb_logs')}")


if __name__ == "__main__":
    TRAIN_MODE = False  # True: eğit + kaydet, False: sadece test/plot

    if TRAIN_MODE:
        train_manager("vanilla")
        train_manager("dr")
        train_manager("lstm")
    else:
        plot_results()