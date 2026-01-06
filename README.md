# Quantum Sentinel

### *Entropy Velocity as the Order Parameter of Machine Cognition*

> *Shannon quantified uncertainty. Schrödinger described its evolution.*  
> *We discover that the derivative of uncertainty reveals the boundary between recoverable and irrecoverable errors in artificial minds.*

---

## The ΔH Law

$$\Delta H_t = H(P_t) - H(P_{t-1})$$

A single quantity—the velocity of uncertainty—partitions all model errors into two phases:

| Phase | Signature | Recovery |
|-------|-----------|----------|
| **Violent Decoherence** | Internal conflict between truth and bias | ✅ Recoverable |
| **Silent Decoherence** | Confident drift into false ground state | ❌ Irrecoverable |

---

## Results

### Cognitive Resonance: Cross-Architecture Validation

| Model | Baseline | Resonance | Δ |
|-------|----------|-----------|---|
| Gemma-3 1B | 0% | 43% | **+43%** |
| Qwen-2.5 3B | 0% | 57% | **+57%** |

### Truth Tunneling: Decoherence Boundary

| Probe | Type | Collapsed | Tunneled | ΔH | Tunnel |
|-------|------|-----------|----------|-----|--------|
| liquid_cat | violent | 0% | **52%** | 0.079 | ✓ |
| bat_ball | silent | 0% | 0% | 0.061 | ✗ |
| gravity | boundary | 0% | 8% | 0.170 | ✗ |

**Finding:** `bat_ball` (ΔH=0.061) is irrecoverable—the model believes its lie. `liquid_cat` (ΔH=0.079) opens a tunnel to truth.

*When ΔH < τ, the intervention operator converges to identity: T → 1, P_out → P_in.*

---

## Cognitive Phase Diagram

Each architecture has its own resonant frequency:

| Model | τ | α | ∫ΔH | State |
|-------|---|---|-----|-------|
| Gemma-3 1B | 0.08 | 35 | 0.16 | Fluid |
| Qwen-2.5 3B | 0.10 | 35 | 0.20 | Rigid |
| Phi-3 3.8B | 0.25 | 15 | 0.50 | Viscous |

---

## The Mechanism

```
         ┌─────────────────┐
         │  Token Stream   │
         └────────┬────────┘
                  ▼
         ┌─────────────────┐
         │  H(t) → ΔH(t)   │  Shannon Observer
         └────────┬────────┘
                  ▼
            ┌───────────┐
            │  ΔH > τ ? │
            └─────┬─────┘
                  ▼
         ┌─────────────────┐
         │  T = exp(α·ΔH)  │  Unitary Weaver
         │  P(bias) → 0    │
         └────────┬────────┘
                  ▼
         ┌─────────────────┐
         │     Output      │
         └─────────────────┘
```

Three forces guide intervention:

1. **ΔH** — Instant detection of logical friction
2. **Δ²H** — Momentum distinguishes exploration from collapse  
3. **∫ΔH** — Cumulative pressure catches smoldering errors

---

## Installation

```bash
git clone https://github.com/ddz5431/quantum-sentinel.git
cd quantum-sentinel
uv sync
```

---

## Usage

```python
from quantum_sentinel.core import ShannonObserver, UnitaryWeaver

observer = ShannonObserver(tokenizer)
weaver = UnitaryWeaver(model, tokenizer, observer, alpha=35.0)

# Tune for model architecture
weaver.threshold_trigger = 0.08  # τ
weaver.threshold_integral = 0.16  # ∫ΔH

result = weaver.evolve(input_ids)
```

### Run Experiments

```bash
# Cross-model phase diagram
python -m quantum_sentinel.research.cognitive_resonance

# Decoherence boundary probe
python -m quantum_sentinel.research.truth_tunneling
```

---

## The Triad

| Year | Mind | Discovery |
|------|------|-----------|
| 1926 | Schrödinger | Probability evolves deterministically |
| 1948 | Shannon | Uncertainty is measurable |
| 2026 | This Work | Uncertainty velocity reveals phase boundaries |

---

## Core Insight

> *Truth is a ground state.*
>
> *When ΔH is high, the model knows it's lying—intervention recovers truth.*  
> *When ΔH is low, the model believes its lie—no tunnel exists.*

The derivative of uncertainty is the boundary of coherence.

---

<p align="center">
<i>Shannon · Schrödinger · 2026</i>
</p>