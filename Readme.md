# PID vs RL for Insulin Dosing in Type 1 Diabetes

**Project:** Reinforcement Learning Course — ENSIAS 2025–2026
**Binôme:** Anass Bensmina / Moubarak Benaqqa
**Deliverable:** Short article + oral presentation

---

## Part 1 — Non-Technical Overview

### What is this project about?

People with Type 1 Diabetes (T1D) can't produce insulin on their own. They need to inject it manually, multiple times a day, to keep their blood sugar (glucose) in a safe range. Inject too little → glucose stays high (long-term damage to eyes, kidneys, heart). Inject too much → glucose drops dangerously low (seizures, coma, death).

The current medical standard is a **manual rule-based system** called Basal-Bolus: a steady background dose plus extra doses around meals. It works but requires the patient to estimate carbohydrates before every meal — a constant cognitive burden, and error-prone.

A natural question: can we replace this rule-based system with an **AI agent that learns the optimal dosing strategy automatically**?

### What are we comparing?

Two approaches to controlling insulin:

1. **PID controller** — a classical engineering control method from the 1940s, used in everything from thermostats to industrial robotics. It reacts to the *current error* (how far glucose is from target), the *accumulated error*, and the *rate of change*. Simple, predictable, well-understood.

2. **Reinforcement Learning (RL) agent** — a neural network that learns by trial and error in simulation. It observes glucose patterns and discovers a dosing strategy by maximizing a "reward" that encourages staying in the safe range and penalizes dangerous lows.

### Why this comparison matters

PID is the current baseline in real medical devices (artificial pancreas systems). If RL can beat PID consistently, that's evidence the medical community should move toward learning-based controllers. If PID holds up, that's also a useful finding — it tells us where RL adds value and where it doesn't.

### What we'll deliver

- A simulator-based experiment comparing PID and an RL agent on virtual T1D patients
- Quantitative results on standard clinical metrics (Time-In-Range, hypoglycemia rate)
- A short article documenting the methodology and findings
- An honest discussion of which method works better, when, and why

---

## Part 2 — Technical Plan

### Stack

- **SimGlucose** (Python, UVA/Padova 2008 simulator, FDA-accepted)
- **Stable-Baselines3** for the RL implementation (PPO)
- **NumPy / Matplotlib / Pandas** for analysis and plotting
- **Custom PID controller** (we implement this ourselves)

### Repository structure

```
insulin-pid-vs-rl/
├── env.py                  # SimGlucose Gymnasium wrapper (already exists)
├── controllers/
│   ├── pid.py              # PID controller implementation
│   └── basal_bolus.py      # Optional: clinical Basal-Bolus baseline
├── train.py                # PPO training script
├── evaluate.py             # Unified evaluation harness (PID + RL)
├── metrics.py              # TIR, LBGI, HBGI, failure rate
├── plots.py                # Glucose trajectories, comparison bar charts
├── results/                # Saved models, CSVs, figures
└── README.md
```

### Step 1 — Fix `env.py` (Day 1)

Issues to fix from previous review:
- Remove unused `alpha`, `beta`, `gamma` constants in `magni_risk`
- Ensure `reset(seed=)` is deterministic
- Verify `obs_raw` unpacking matches current SimGlucose gymnasium API
- Add a `terminate_on_extreme` flag (terminate episode if glucose < 40 or > 600)

### Step 2 — Implement PID controller (Day 2)

```python
class PIDController:
    def __init__(self, kp, ki, kd, target=120, basal_rate=...):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.target = target
        self.basal_rate = basal_rate  # patient-specific
        self.integral = 0
        self.prev_error = 0

    def reset(self):
        self.integral = 0
        self.prev_error = 0

    def get_action(self, glucose):
        error = glucose - self.target
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error

        correction = self.kp * error + self.ki * self.integral + self.kd * derivative
        insulin = max(0, self.basal_rate + correction)
        return insulin
```

Tune `(kp, ki, kd)` per patient using grid search on a few training episodes.
Starting point from the literature: `kp=0.001, ki=0.00001, kd=0.01`. Adjust empirically.

### Step 3 — Train RL agent (Days 3–5)

Use Stable-Baselines3 PPO with the existing `InsulinEnv`:

```python
from stable_baselines3 import PPO
from env import InsulinEnv

env = InsulinEnv(patient_name='adult#001', seed=42)
model = PPO('MlpPolicy', env, verbose=1,
            learning_rate=3e-4, n_steps=2048,
            batch_size=64, n_epochs=10)
model.learn(total_timesteps=200_000)
model.save('results/ppo_adult001')
```

Train one PPO model per patient (3 adults, optionally 3 adolescents).
Budget: 200k steps × 6 patients ≈ a few hours on CPU.

### Step 4 — Evaluation protocol (Day 6)

Unified evaluation for both PID and RL on the same patients, same meal scenarios:

- **Patients:** 3 adults + 3 adolescents (`adult#001-003`, `adolescent#001-003`)
- **Episode length:** 24 hours (288 steps × 5 min)
- **Meal protocol:** breakfast 40g @ 08:00, lunch 80g @ 13:00, dinner 60g @ 20:00 (matches G2P2C evaluation)
- **Trials:** 50 evaluation episodes per (patient, method) pair with different random seeds
- **Random init glucose:** between 110–130 mg/dL

### Step 5 — Metrics (Day 6)

Implement these in `metrics.py`:

| Metric | Definition | Goal |
|---|---|---|
| **TIR** | % time glucose ∈ [70, 180] mg/dL | maximize |
| **TBR** | % time glucose < 70 mg/dL (hypoglycemia) | minimize |
| **TAR** | % time glucose > 180 mg/dL (hyperglycemia) | minimize |
| **LBGI** | Low Blood Glucose Index (Magni) | minimize |
| **HBGI** | High Blood Glucose Index (Magni) | minimize |
| **Failure rate** | % episodes with glucose < 40 or > 600 | minimize |
| **Mean glucose** | average glucose over episode | target ≈ 120 |
| **CV** | glucose coefficient of variation | minimize |

### Step 6 — Statistical comparison (Day 7)

For each metric, run Mann-Whitney U test (PID vs RL) per patient and report:
- Median + IQR for each method
- p-value
- Effect size (Cohen's r)

### Step 7 — Plots (Day 7)

- Box plots of TIR per patient (PID vs RL side by side)
- Sample 24h glucose trajectories (PID and RL on the same patient, same seed)
- Failure rate bar chart
- Cumulative reward curve during RL training (sanity check)

### Step 8 — Article (Days 8–10)

Structure:
1. **Introduction** — T1D, glucose control, motivation
2. **Background** — MDP formulation, PID vs RL conceptually
3. **Methodology** — env setup, PID tuning, PPO config, evaluation protocol
4. **Results** — tables, plots, statistical tests
5. **Discussion** — honest analysis: where RL wins, where PID wins, limitations
6. **Conclusion + future work**

Target length: 6–8 pages, LaTeX, IEEE or Elsevier template.

---

## Part 3 — Realistic Expectations

### What we will likely find

Based on literature:
- **PID will be competitive on stable, predictable meal scenarios** — it's a well-tuned reactive controller
- **RL will likely beat PID on TIR but may have a higher failure rate** without careful reward shaping
- **PID may overreact to meals** because it can't anticipate them
- **RL may struggle on adolescents** (higher physiological variability)

### What "good results" means here

We are NOT trying to beat G2P2C or publish at Nature. Good results means:
1. Both methods are correctly implemented
2. The experimental protocol is clean (same patients, same meals, same metrics)
3. The conclusion is honest and supported by the data
4. We can defend the methodology in oral

### Risk mitigation

- **Risk:** RL doesn't learn anything useful in 200k steps → **Mitigation:** use 3 random seeds, increase to 500k if needed, log learning curves daily
- **Risk:** PID is hard to tune → **Mitigation:** use literature-reported parameters as starting point, do grid search on 1 patient first
- **Risk:** SimGlucose installation issues → **Mitigation:** verify install on day 1, use `pip install simglucose` and run the sanity check in `env.py`

---

## Part 4 — Division of Work (Binôme)

| Task | Owner | Estimate |
|---|---|---|
| `env.py` fixes + sanity tests | A | 1 day |
| PID implementation + tuning | A | 2 days |
| PPO training infrastructure | B | 2 days |
| Evaluation harness + metrics | B | 1 day |
| Run all experiments | both | 1 day |
| Analysis + plots | both | 1 day |
| Article draft | both (split sections) | 2 days |
| Polish + submit | both | 1 day |

**Total: ~10 working days**

---

## Part 5 — Deliverables Checklist

- [ ] Working code in a GitHub repo with README
- [ ] Trained PPO models for all evaluated patients (saved as `.zip`)
- [ ] Results CSVs (per-episode metrics for both methods)
- [ ] Figures (box plots, trajectories, training curves)
- [ ] Article (PDF, ~6–8 pages)
- [ ] Slides for oral presentation
- [ ] Honest "limitations" section

---

## References to cite

1. Hettiarachchi et al. (2024) — G2P2C paper (your direct reference for benchmark methodology)
2. Magni et al. (2011) — risk index used in the reward
3. Man et al. (2014) — UVA/Padova simulator
4. Steil (2013) — PID for closed-loop artificial pancreas
5. Schulman et al. (2017) — PPO
6. Marling & Bunescu (2020) — OhioT1DM (mention even if not used, for context)
