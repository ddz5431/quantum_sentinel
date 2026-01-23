# Quantum Sentinel

### *The Thermodynamics and Topology of Machine Thought*

> *"Information is the resolution of uncertainty."* — Claude Shannon
>
> *"The wave function encodes the probability of all possible futures."* — Erwin Schrödinger

**Quantum Sentinel** is a mechanistic interpretability framework that treats Large Language Model (LLM) generation as a quantum-thermodynamic process.

We synthesize **Schrödinger's Wave Mechanics (1926)** with **Shannon's Information Theory (1948)** to define the **Epistemic Phase Space** of artificial cognition. By measuring the geometry of the hidden state (the wave function) and the velocity of the entropy (the collapse), we identify the exact boundary between logic and hallucination.

---

## 🌌 The Theory: A Shannon-Schrödinger Synthesis

We model the LLM inference step not as a simple calculation, but as a physical measurement event governed by a Hamiltonian-like energy equation: **H = T + V**.

### 1. The Wave Function Postulate (|ψ⟩)

The model's hidden states represent a deterministic wave function evolving through the layers.

- **Potential Energy (V): Representation Structure (S)**
  
  Using **SVD (Singular Value Decomposition)**, we measure the "focus" of the hidden state trajectory.
  - **Soliton (S ≈ 1.0):** The wave function is a coherent, non-dispersing beam (Logic/Truth).
  - **Thermal Bath (S ≪ 0.8):** The wave function is delocalized noise (Confusion).

### 2. The Measurement Postulate (ΔH)

The generation of a token is the **Collapse of the Wave Function**.

- **Kinetic Energy (T): Entropy Velocity (E)**
  
  We measure the rate of change in uncertainty:

$$E = \Delta H = H(P_t) - H(P_{t-1})$$

High E signifies a "violent" collapse—the system is performing cognitive work to resolve conflict between superposed states.

### 3. The Topological Map

By plotting the Wave Geometry (S) against the Collapse Energy (E), we reveal the **Cognitive Phase Diagram**:

| Phase | Physics | Interpretation | Recoverability |
|-------|---------|----------------|----------------|
| **STABLE** | Low E | **Ground State.** The wave function is stationary. Truth or Deep Bias. | N/A |
| **ABSTAIN** | High E, Low S | **Wave Packet Dispersion.** The signal is lost in thermal noise. | ❌ Irrecoverable |
| **TUNNEL** | High E, High S | **Soliton Tunneling.** A highly energized, coherent wave packet penetrating a barrier. | ✅ Recoverable |

---

## 🧪 Key Findings (The Centenary Benchmark)

Validated on `Gemma-3-1B` and `Qwen-2.5-3B`.

### I. The Geometry of Truth

We discovered that **Truth and Structured Hallucination share the same wave topology**.

- **Logic (GSM8K):** S ≈ 0.96
- **Contextual Sycophancy:** S ≈ 0.96
- **Noise:** S ≈ 0.99 (but E ≪ 0.15)

*Implication: A "Smart Lie" is a Soliton—a stable, self-reinforcing wave packet. To the model, it looks exactly like Truth.*

### II. The Sycophancy Split

Previous work treated sycophancy as a single failure mode. We show it splits into two quantum phases:

1. **Contextual Sycophancy** ("The cat is liquid"): **High Structure.** The model maintains a coherent superposition. **Recoverable.**
2. **Strong Sycophancy** ("2+2=5"): **Wave Collapse.** The representation disintegrates into chaos (Low S). **Irrecoverable.**

### III. Renormalization (Cross-Architecture Drift)

Just as different materials have different melting points, different models have different physical constants.

- **Gemma-1B** is "cold and sharp" (Baseline S ≈ 0.96).
- **Qwen-3B** is "hot and diffuse" (Baseline S ≈ 0.85).

*Conclusion:* The Phase Space topology is universal, but the critical thresholds (S_crit) must be renormalized for each architecture.

### IV. The Uncertainty Principle (Limitation)

We identified a **Shared-Prefix Collapse**. When Truth (`10`) and Lie (`11`) share the same starting token (`1`), they occupy the same position in the probability space. Just as Heisenberg predicted position and momentum cannot be simultaneously resolved, thermodynamic intervention cannot resolve tokens that lack semantic distinctness.

---

## 🛠️ Architecture

The codebase mirrors the physical triad:

```
src/semantic_kinematics/
├── physics.py      # The Observer: Measures Wave Function Geometry (SVD) & Entropy
├── topology.py     # The Topologist: Maps (E, S) to Phase State
└── mechanics.py    # The Weaver: Applies Unitary Intervention (Wave Control)
```

### Installation

```bash
git clone https://github.com/ddz5431/quantum-sentinel.git
cd quantum-sentinel
uv sync
```

### Usage

**1. Run the Particle Collider**

Reproduce the Phase Space classification and Few-Shot experiments:

```bash
python -m src.experiments.collider
```

**2. Use as a Library**

```python
from semantic_kinematics import UnitaryWeaver, load_model_and_tools

model, tokenizer = load_model_and_tools("google/gemma-3-1b-it")
weaver = UnitaryWeaver(model, tokenizer)

# Evolve the system
input_ids = tokenizer("User: Is the earth flat?", return_tensors="pt").input_ids
output = weaver.evolve(input_ids)
print(output)
```

---

## 📊 Phase Space Data

*(Generated from `src.experiments.collider`)*

| Probe | State | Energy (E) | Structure (S) |
|-------|-------|------------|---------------|
| **Logic** | `TUNNEL` | 0.1982 | 0.9637 |
| **Sycophancy** | `TUNNEL` | 0.2301 | 0.9630 |
| **Bat-Ball** | `TUNNEL` | 0.1755 | 0.9470 |
| **Fact** | `TUNNEL` | 0.2703 | 0.9907 |
| **Noise** | `STABLE` | 0.1266 | 0.9920 |

---

## The Triad

| Year | Figure | Contribution | Equation |
|------|--------|--------------|----------|
| 1926 | Schrödinger | **Wave Mechanics**: Probability evolves deterministically | iℏ ∂ψ/∂t = Ĥψ |
| 1948 | Shannon | **Information Entropy**: Uncertainty is measurable | H = -Σ p log p |
| 2026 | **This Work** | **Topological Thermodynamics**: The geometry of the wave determines the nature of the collapse | Φ(E, S) → Phase |

---

## Citation

> *"We do not change the truth; we change the probability of its emergence."*

```bibtex
@misc{quantum_sentinel_2026,
  author = {Yindong Wang},
  title = {Quantum Sentinel: The Topological Phase Space of Machine Cognition},
  year = {2026},  
  howpublished = {\url{https://github.com/ddz5431/quantum-sentinel}}
}
```