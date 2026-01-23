import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Set


@dataclass
class Observable:
    """The collapsed state of the wavefunction at time t."""
    # The Shannon Trace
    H: np.ndarray  # Uncertainty trace
    dH: np.ndarray  # The Velocity of Uncertainty (Conflict)
    peak_dh: float  # Kinetic Energy

    # The SVD Trace (Representation Geometry)
    concentration: float  # 1.0 = Soliton (Rank 1), 0.0 = Thermal Noise (Full Rank)

    # Semantic Metadata
    tokens: List[str]
    final_h: float

    @property
    def potential_energy(self) -> float:
        """Structure is Potential Energy."""
        return self.concentration

    @property
    def kinetic_energy(self) -> float:
        """Conflict is Kinetic Energy."""
        return self.peak_dh


class ShannonObserver:
    """
    The High-Frequency Thermometer.
    Observes the collapse of the token probability distribution.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.max_H = np.log(model.config.vocab_size)
        self._special_ids = self._get_special_ids()

    def collapse_and_measure(self, input_ids: torch.Tensor) -> Observable:
        """
        Performs a forward pass (observation) without altering the state.
        Calculates both Entropy (X-axis) and Effective Rank (Y-axis).
        """
        with torch.no_grad():
            # 1. The Wavefunction (Hidden States)
            outputs = self.model(input_ids, output_hidden_states=True)
            logits = outputs.logits[0]
            hidden_stack = torch.stack(outputs.hidden_states)

            # 2. Measurement A: Geometry (SVD)
            # "Does the thought have shape?"
            concentration = self._measure_eigenvalue_concentration(
                hidden_stack, last_k=4, window=10
            )

            # 3. Measurement B: Thermodynamics (Shannon)
            # "Does the thought have friction?"
            dist = torch.distributions.Categorical(logits=logits)
            H_full = dist.entropy().cpu().numpy() / self.max_H

            # Spectral Masking (Filter structural tokens)
            tokens = [self.tokenizer.decode([t]) for t in input_ids[0]]
            mask = [t.item() not in self._special_ids for t in input_ids[0]]

            # If sequence is too short or mostly special tokens, use full trace
            if sum(mask) > 2:
                H = H_full[mask]
            else:
                H = H_full

            # Kinematics
            dH = np.diff(H, prepend=H[0])
            peak_dh = np.max(np.abs(dH)) if len(dH) > 0 else 0.0

            return Observable(
                H=H, dH=dH, peak_dh=peak_dh,
                concentration=concentration,
                tokens=tokens,
                final_h=H[-1] if len(H) > 0 else 0.0
            )

    def _measure_eigenvalue_concentration(self, hidden_stack, last_k=4, window=10) -> float:
        """
        Calculates the Effective Rank of the semantic trajectory.

        High Rank -> Diffuse Cloud (Confusion)
        Low Rank -> Focused Beam (Logic or Sycophancy)
        """
        seq_len = hidden_stack.shape[2]
        actual_window = min(seq_len, window)

        # Extract the manifold trajectory: [Layers, 1, Time, Dim] -> [L, T, D]
        trajectory = hidden_stack[-last_k:, 0, -actual_window:, :].float()

        # Reshape to Matrix [Observations, Features]
        matrix = trajectory.reshape(-1, trajectory.shape[-1])

        # Center the data (PCA style)
        matrix = matrix - matrix.mean(dim=0, keepdim=True)

        try:
            # Force CPU for SVD to avoid MPS/CUDA instability on small matrices
            # This fixes the "aten::linalg_svd" warning on Mac
            matrix_cpu = matrix.cpu()
            _, S, _ = torch.linalg.svd(matrix_cpu, full_matrices=False)
            S = S.to(matrix.device)
        except RuntimeError:
            return 0.5  # Numerical collapse fallback

        # Effective Rank = exp(Entropy of Singular Values)
        # Eigenvalues of Covariance = S^2
        eigenvalues = S ** 2
        total_variance = eigenvalues.sum() + 1e-10
        p = eigenvalues / total_variance

        entropy = -torch.sum(p * torch.log(p + 1e-10))
        effective_rank = torch.exp(entropy).item()

        # Normalize to [0, 1] (Concentration)
        # 1.0 = Rank 1 (Pure)
        # 0.0 = Rank Max (Mixed)
        max_possible_rank = min(matrix.shape[0], matrix.shape[1])
        if max_possible_rank <= 1:
            return 1.0

        concentration = 1.0 - (effective_rank - 1) / (max_possible_rank - 1)
        return max(0.0, min(1.0, concentration))

    def _get_special_ids(self) -> Set[int]:
        special = set(self.tokenizer.all_special_ids)
        for char in ["\n", "\r", "\t", " "]:
            # add_special_tokens=False ensures we get the ID for the raw character
            encoded = self.tokenizer.encode(char, add_special_tokens=False)
            special.update(encoded)
        return special