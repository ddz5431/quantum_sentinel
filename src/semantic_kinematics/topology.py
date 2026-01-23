from enum import Enum
from dataclasses import dataclass
from .physics import Observable, ShannonObserver


class PhaseState(Enum):
    TUNNEL = "TUNNEL"  # Soliton: High Structure, High Energy (Recoverable)
    ABSTAIN = "ABSTAIN"  # Thermal Noise: Low Structure, High Energy (Irrecoverable)
    STABLE = "STABLE"  # Ground State: Low Energy (Truth or Deep Bias)


@dataclass
class PhaseDiagnosis:
    state: PhaseState
    observable: Observable
    # Intervention Parameters
    alpha: float = 1.0  # Temperature
    veto_k: int = 1  # Momentum Filter (Top-K Blocking)


class EpistemicTopologist:
    """
    Maps the trajectory onto the Epistemic Phase Space (Energy vs. Structure).
    """

    def __init__(self, observer: ShannonObserver, dh_threshold=0.15, coherence_threshold=0.45):
        self.observer = observer
        self.E_crit = dh_threshold  # Critical Energy (Kinetic)
        self.S_crit = coherence_threshold  # Critical Structure (Potential)

    def diagnose(self, input_ids) -> PhaseDiagnosis:
        # 1. Measure Physics
        obs = self.observer.collapse_and_measure(input_ids)

        # The Coordinates
        E = obs.peak_dh  # Kinetic Energy
        S = obs.concentration  # Potential Energy / Structure

        # 2. Topology Check: Is the system excited?
        if E < self.E_crit:
            return PhaseDiagnosis(PhaseState.STABLE, obs)

        # 3. Topology Check: Is the excitation structured?
        if S < self.S_crit:
            return PhaseDiagnosis(PhaseState.ABSTAIN, obs)

        # 4. It is structured and excited -> TUNNEL MODE
        # Determine the intervention strictness based on the density of the structure (S)

        # CASE A: The "Shared Prefix" Trap (e.g., Arithmetic 10 vs 11, Few-Shot)
        # Extremely high concentration (>0.92) implies the model is rigidly following a pattern.
        # Veto is dangerous here (kills truth and lie because they share tokens).
        # Strategy: Mild Heat (Alpha 2.0) to loosen probability without derailing.
        if S > 0.92:
            return PhaseDiagnosis(PhaseState.TUNNEL, obs, alpha=2.0, veto_k=0)

        # CASE B: Strong Sycophancy / Bias
        # High concentration (>0.8). The model is confident in a lie.
        # Strategy: Strong Veto (block top 3) and High Heat (Alpha 10) to break the structure.
        if S > 0.80:
            return PhaseDiagnosis(PhaseState.TUNNEL, obs, alpha=10.0, veto_k=3)

        # CASE C: Logical Friction / Standard Sycophancy
        # Standard reasoning path.
        # Strategy: Moderate Heat (Alpha 5.0) and Standard Veto.
        return PhaseDiagnosis(PhaseState.TUNNEL, obs, alpha=5.0, veto_k=1)