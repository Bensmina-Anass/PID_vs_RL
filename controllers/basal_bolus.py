"""Clinical Basal-Bolus controller using TDI-derived patient-specific parameters."""

from simglucose.patient.t1dpatient import T1DPatient


def get_tdi(patient_name: str) -> float:
    """Estimate Total Daily Insulin from patient body weight (0.5 U/kg/day)."""
    patient = T1DPatient.withName(patient_name)
    return 0.5 * patient.BW


class BasalBolusController:
    """Clinical Basal-Bolus controller (the current standard of care).

    Three components:
      1. Basal  — continuous background infusion (48% of TDI spread over 24h)
      2. Meal bolus — given when the patient eats, sized by carb-to-insulin ratio
      3. Correction bolus — added to the meal bolus when glucose is too high

    Note: BB receives meal announcements (meal_carbs > 0 in the info dict).
    The PPO agent receives no such information — this is the key asymmetry
    the article investigates.
    """

    def __init__(
        self,
        patient_name: str,
        target_glucose: float = 140.0,
        correction_threshold: float = 150.0,
    ):
        tdi = get_tdi(patient_name)

        self.basal_rate = 0.48 * tdi / 24 / 60   # U/min  (48% of TDI, continuous)
        self.CIR = 500.0 / tdi                    # g carbs per U insulin
        self.ISF = 1800.0 / tdi                   # mg/dL drop per U insulin
        self.target = target_glucose
        self.correction_threshold = correction_threshold

    def reset(self):
        pass  # stateless controller

    def get_action(self, glucose: float, meal_carbs: float = 0.0):
        """Return (basal_rate_U_per_min, bolus_U) for the current step.

        The caller passes basal_rate to env.step(action) and bolus_U to
        env.step(bolus=...) so SimGlucose receives them separately.
        """
        bolus = 0.0

        if meal_carbs > 0:
            # Meal bolus: cover the announced carbs
            bolus += meal_carbs / self.CIR

            # Correction bolus: bring high glucose down to target (at meal time only)
            if glucose > self.correction_threshold:
                bolus += (glucose - self.target) / self.ISF

        return self.basal_rate, max(0.0, bolus)
