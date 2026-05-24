"""SimGlucose Gymnasium wrapper for closed-loop insulin delivery."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from datetime import datetime
from simglucose.simulation.env import T1DSimEnv as _SimGlucoseEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.controller.base import Action

# Fixed 24h meal protocol: minute-of-day -> grams of carbs
MEAL_SCHEDULE = {
    480:  40,   # 08:00 — breakfast
    780:  80,   # 13:00 — lunch
    1200: 60,   # 20:00 — dinner
}

PATIENTS = [
    'adult#001', 'adult#002', 'adult#003',
    'adolescent#001', 'adolescent#002', 'adolescent#003',
]

MAX_STEPS = 288   # 24 h at 5-min intervals
MAX_BASAL = 0.1   # U/min — PPO action ceiling


def magni_risk(glucose: float) -> float:
    """Symmetric risk index from Magni et al. (2011)."""
    f = 1.509 * (np.log(float(np.clip(glucose, 1, None))) ** 1.084 - 5.381)
    return 10.0 * f ** 2


class _FixedMealScenario:
    """Passes the fixed meal schedule to the SimGlucose internal engine."""

    def __init__(self, start_time: datetime):
        self.start_time = start_time

    def get_action(self, t):
        minutes = int(round((t - self.start_time).total_seconds() / 60))
        cho = MEAL_SCHEDULE.get(minutes, 0)
        # SimGlucose scenarios return Action(basal=0, bolus=CHO_grams)
        return Action(basal=0, bolus=cho)


class InsulinEnv(gym.Env):
    """Gymnasium environment wrapping the SimGlucose UVA/Padova simulator.

    Observation (3 floats): [CGM glucose, sin(time_of_day), cos(time_of_day)]
    Action (1 float):       basal insulin rate in U/min — [0, MAX_BASAL]
    info dict:              {'meal_carbs': float, 'glucose': float}

    The meal_carbs field in info lets the Basal-Bolus controller know what
    meal is happening at the current step. The PPO agent never sees meals.
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        patient_name: str = 'adult#001',
        seed: int = 0,
        terminate_on_extreme: bool = True,
    ):
        super().__init__()
        assert patient_name in PATIENTS, f'Unknown patient: {patient_name}'
        self.patient_name = patient_name
        self.seed_val = seed
        self.terminate_on_extreme = terminate_on_extreme

        self.action_space = spaces.Box(
            low=np.float32(0.0), high=np.float32(MAX_BASAL), shape=(1,)
        )
        # Obs: [glucose, sin(time), cos(time)] — no meal info (PPO must infer)
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([600.0, 1.0, 1.0], dtype=np.float32),
        )

        self._sim = None
        self._step_count = 0
        self._glucose = 120.0

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        s = seed if seed is not None else self.seed_val
        self._sim = self._build_sim(s)
        self._step_count = 0
        # One warm-up step to get the initial CGM reading
        bg = self._advance_sim(rate=0.0, bolus=0.0)
        self._glucose = bg
        return self._obs(), {}

    def step(self, action, bolus: float = 0.0):
        """Advance one 5-minute step.

        Parameters
        ----------
        action : array-like or float
            Basal insulin rate in U/min (PPO agent output).
        bolus : float
            One-time bolus in U, used by the Basal-Bolus controller only.
            Defaults to 0 for PPO.
        """
        rate = float(np.clip(
            action[0] if hasattr(action, '__len__') else action,
            0.0, MAX_BASAL,
        ))
        bolus = float(np.clip(bolus, 0.0, 30.0))

        # Announce the meal that starts at this step (BB reads from info)
        meal_carbs = float(MEAL_SCHEDULE.get(self._step_count * 5, 0))

        bg = self._advance_sim(rate, bolus)
        self._glucose = bg
        self._step_count += 1

        reward = self._reward(bg)
        info = {'meal_carbs': meal_carbs, 'glucose': bg}

        terminated = (
            self.terminate_on_extreme and (bg < 40 or bg > 600)
        )
        if terminated:
            reward -= 100.0

        truncated = self._step_count >= MAX_STEPS

        return self._obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_sim(self, seed: int) -> _SimGlucoseEnv:
        np.random.seed(seed)
        patient = T1DPatient.withName(self.patient_name)
        sensor = CGMSensor.withName('Dexcom', seed=seed)
        pump = InsulinPump.withName('Insulet')
        scenario = _FixedMealScenario(datetime(2025, 1, 1, 0, 0, 0))
        return _SimGlucoseEnv(patient, sensor, pump, scenario)

    def _advance_sim(self, rate: float, bolus: float) -> float:
        result = self._sim.step(Action(basal=rate, bolus=bolus))
        # SimGlucose raw env returns (bg, reward, done, info) — 4-tuple
        return float(result[0])

    def _obs(self) -> np.ndarray:
        angle = 2.0 * np.pi * (self._step_count * 5) / 1440.0
        return np.array(
            [self._glucose, np.sin(angle), np.cos(angle)],
            dtype=np.float32,
        )

    def _reward(self, glucose: float) -> float:
        tir_bonus = 1.0 if 70.0 <= glucose <= 180.0 else 0.0
        return tir_bonus - 0.1 * magni_risk(glucose)
