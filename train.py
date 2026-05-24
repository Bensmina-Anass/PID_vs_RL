"""Train one PPO agent per patient and save to results/."""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from env import InsulinEnv

PATIENTS = [
    'adult#001', 'adult#002', 'adult#003',
    'adolescent#001', 'adolescent#002', 'adolescent#003',
]

TOTAL_TIMESTEPS = 200_000
RESULTS_DIR = 'results'


def train_patient(patient_name: str, seed: int = 42):
    print(f'\n=== Training PPO for {patient_name} ===')

    env = make_vec_env(
        lambda: InsulinEnv(patient_name=patient_name, seed=seed),
        n_envs=1,
    )

    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        seed=seed,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    slug = patient_name.replace('#', '')
    save_path = os.path.join(RESULTS_DIR, f'ppo_{slug}')
    model.save(save_path)
    print(f'Saved to {save_path}.zip')
    env.close()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for patient in PATIENTS:
        train_patient(patient)


if __name__ == '__main__':
    main()
