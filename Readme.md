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

