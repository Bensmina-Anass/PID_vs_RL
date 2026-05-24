"""Generate all figures for the paper."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from env import InsulinEnv, PATIENTS
from controllers.basal_bolus import BasalBolusController

RESULTS_DIR = 'results'
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')

METHODS  = ['bb', 'ppo']
COLORS   = {'bb': '#4CAF50', 'ppo': '#F44336'}
LABELS   = {'bb': 'Basal-Bolus (clinical)', 'ppo': 'PPO (RL)'}


# ------------------------------------------------------------------
# Figure 1: TIR box plots per patient
# ------------------------------------------------------------------

def tir_boxplots(df: pd.DataFrame):
    patients = df['patient'].unique()
    methods = [m for m in METHODS if m in df['method'].unique()]

    fig, axes = plt.subplots(1, len(patients), figsize=(3.5 * len(patients), 5), sharey=True)
    if len(patients) == 1:
        axes = [axes]

    for ax, patient in zip(axes, patients):
        data = [
            df[(df['patient'] == patient) & (df['method'] == m)]['tir'].values
            for m in methods
        ]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5)
        for patch, m in zip(bp['boxes'], methods):
            patch.set_facecolor(COLORS[m])
            patch.set_alpha(0.8)
        ax.set_title(patient, fontsize=9)
        ax.set_xticks(range(1, len(methods) + 1))
        ax.set_xticklabels([LABELS[m] for m in methods], fontsize=8, rotation=15)
        ax.set_ylim(0, 105)
        ax.axhline(70, color='grey', linewidth=0.8, linestyle='--', label='Clinical target')

    axes[0].set_ylabel('Time-In-Range (%)')
    fig.suptitle('Time-In-Range [70–180 mg/dL] by Patient and Method', fontsize=11)
    fig.tight_layout()
    _save(fig, 'tir_boxplots.png')


# ------------------------------------------------------------------
# Figure 2: 24h glucose trajectories
# ------------------------------------------------------------------

def glucose_trajectory(patient_name: str, seed: int = 0):
    slug = patient_name.replace('#', '')
    traces = {}

    # Basal-Bolus
    bb = BasalBolusController(patient_name=patient_name)
    env = InsulinEnv(patient_name=patient_name, seed=seed)
    obs, info = env.reset(seed=seed)
    bb.reset()
    trace = [info.get('glucose', obs[0])]
    done = False
    while not done:
        basal, bolus = bb.get_action(obs[0], info.get('meal_carbs', 0.0))
        obs, _, terminated, truncated, info = env.step(basal, bolus=bolus)
        trace.append(info['glucose'])
        done = terminated or truncated
    traces['bb'] = trace

    # PPO (if model exists)
    model_path = os.path.join(RESULTS_DIR, f'ppo_{slug}.zip')
    if os.path.exists(model_path):
        model = PPO.load(model_path)
        env = InsulinEnv(patient_name=patient_name, seed=seed)
        obs, info = env.reset(seed=seed)
        trace = [info.get('glucose', obs[0])]
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            trace.append(info['glucose'])
            done = terminated or truncated
        traces['ppo'] = trace

    fig, ax = plt.subplots(figsize=(11, 4))
    for m, trace in traces.items():
        t = np.arange(len(trace)) * 5 / 60  # convert steps → hours
        ax.plot(t, trace, label=LABELS[m], color=COLORS[m], linewidth=1.5)

    ax.axhspan(70, 180, alpha=0.08, color='green')
    ax.axhline(70,  color='orange', linestyle='--', linewidth=0.9)
    ax.axhline(180, color='orange', linestyle='--', linewidth=0.9)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Glucose (mg/dL)')
    ax.set_title(f'24h Glucose Trajectory — {patient_name}  (seed={seed})')
    ax.legend()
    fig.tight_layout()
    _save(fig, f'trajectory_{slug}.png')


# ------------------------------------------------------------------
# Figure 3: Failure rate bar chart
# ------------------------------------------------------------------

def failure_rate_chart(df: pd.DataFrame):
    patients = df['patient'].unique()
    methods = [m for m in METHODS if m in df['method'].unique()]

    x = np.arange(len(patients))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))

    for i, m in enumerate(methods):
        rates = [
            100.0 * df[(df['patient'] == p) & (df['method'] == m)]['failure'].mean()
            for p in patients
        ]
        ax.bar(x + i * width, rates, width, label=LABELS[m], color=COLORS[m], alpha=0.85)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(patients, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('Episode failure rate (%)')
    ax.set_title('Failure Rate (glucose < 40 or > 600 mg/dL) by Patient and Method')
    ax.legend()
    fig.tight_layout()
    _save(fig, 'failure_rate.png')


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _save(fig, filename: str):
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Saved {path}')


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, 'results.csv')
    if not os.path.exists(csv_path):
        print(f'No results.csv found — run evaluate.py first.')
        return

    df = pd.read_csv(csv_path)
    tir_boxplots(df)
    failure_rate_chart(df)
    for patient in PATIENTS:
        glucose_trajectory(patient_name=patient, seed=0)


if __name__ == '__main__':
    main()
