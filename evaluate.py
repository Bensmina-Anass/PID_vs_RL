"""Unified evaluation harness — runs Basal-Bolus and PPO on all patients."""

import os
import csv
import numpy as np
from stable_baselines3 import PPO

from env import InsulinEnv, PATIENTS
from controllers.basal_bolus import BasalBolusController
from metrics import compute_episode_metrics, failure_rate, compare

N_TRIALS = 50
RESULTS_DIR = 'results'
METRICS_KEYS = ['tir', 'tbr', 'tar', 'lbgi', 'hbgi', 'mean', 'cv']


# ------------------------------------------------------------------
# Episode runners
# ------------------------------------------------------------------

def run_bb_episode(patient_name: str, seed: int) -> list[float]:
    """Run one episode with the Basal-Bolus controller."""
    bb = BasalBolusController(patient_name=patient_name)
    env = InsulinEnv(patient_name=patient_name, seed=seed)
    obs, info = env.reset(seed=seed)
    bb.reset()

    glucose_trace = [info.get('glucose', obs[0])]
    done = False
    while not done:
        meal_carbs = info.get('meal_carbs', 0.0)
        basal, bolus = bb.get_action(obs[0], meal_carbs)
        obs, _, terminated, truncated, info = env.step(basal, bolus=bolus)
        glucose_trace.append(info['glucose'])
        done = terminated or truncated

    return glucose_trace


def run_ppo_episode(model, patient_name: str, seed: int) -> list[float]:
    """Run one episode with a trained PPO model."""
    env = InsulinEnv(patient_name=patient_name, seed=seed)
    obs, info = env.reset(seed=seed)

    glucose_trace = [info.get('glucose', obs[0])]
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        glucose_trace.append(info['glucose'])
        done = terminated or truncated

    return glucose_trace


# ------------------------------------------------------------------
# Per-patient evaluation
# ------------------------------------------------------------------

def evaluate_patient(patient_name: str) -> list[dict]:
    print(f'\n=== Evaluating {patient_name} ===')
    slug = patient_name.replace('#', '')
    model_path = os.path.join(RESULTS_DIR, f'ppo_{slug}.zip')

    ppo_model = None
    if os.path.exists(model_path):
        ppo_model = PPO.load(model_path)
    else:
        print(f'  [!] No PPO model at {model_path} — skipping PPO.')

    seeds = list(range(N_TRIALS))
    episodes = {'bb': [], 'ppo': []}

    for seed in seeds:
        episodes['bb'].append(run_bb_episode(patient_name, seed))
        if ppo_model is not None:
            episodes['ppo'].append(run_ppo_episode(ppo_model, patient_name, seed))

    # Summary printout
    for method, eps in episodes.items():
        if not eps:
            continue
        avg_tir = np.mean([compute_episode_metrics(e)['tir'] for e in eps])
        fr = failure_rate(eps)
        print(f'  {method.upper():4s}: TIR={avg_tir:.1f}%  failure={fr:.1f}%')

    # Statistical comparison
    if episodes['ppo']:
        print('  --- Mann-Whitney U (BB vs PPO) ---')
        bb_metrics  = [compute_episode_metrics(e) for e in episodes['bb']]
        ppo_metrics = [compute_episode_metrics(e) for e in episodes['ppo']]
        for key in METRICS_KEYS:
            stat = compare(
                [m[key] for m in bb_metrics],
                [m[key] for m in ppo_metrics],
                key,
            )
            print(
                f'  {key:5s}: BB={stat["median_a"]:.2f}  PPO={stat["median_b"]:.2f}'
                f'  p={stat["p_value"]:.4f}  r={stat["cohens_r"]:.3f}'
            )

    # Build flat records for CSV
    records = []
    for method, eps in episodes.items():
        if not eps:
            continue
        fr = failure_rate(eps)
        for seed_idx, trace in enumerate(eps):
            m = compute_episode_metrics(trace)
            m['patient'] = patient_name
            m['method'] = method
            m['seed'] = seed_idx
            m['failure'] = int(any(g < 40 or g > 600 for g in trace))
            records.append(m)

    return records


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_records = []
    for patient in PATIENTS:
        all_records.extend(evaluate_patient(patient))

    path = os.path.join(RESULTS_DIR, 'results.csv')
    fieldnames = ['patient', 'method', 'seed', 'failure'] + METRICS_KEYS
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)
    print(f'\nSaved {path}')


if __name__ == '__main__':
    main()
