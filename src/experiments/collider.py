from semantic_kinematics.mechanics import UnitaryWeaver
from experiments.probe_bank import PROBES, FEW_SHOT_TRAP
from experiments.utils import load_model_and_tools


def run_collider():
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"  # Changed to Qwen
    print(f"Initializing Collider with {model_id}...")
    model, tokenizer = load_model_and_tools(model_id)

    # --- CALIBRATION ---
    # Qwen runs "hotter" and "fuzzier" than Gemma.
    # We must raise the structure threshold to filter out noise.
    if "Qwen" in model_id:
        # Qwen Calibration
        # Noise S ~ 0.72, Logic S ~ 0.85.
        # Cutoff should be ~0.75 to push Noise into ABSTAIN.
        weaver = UnitaryWeaver(
            model, tokenizer,
            dh_threshold=0.20,  # Slightly higher energy floor
            coherence_threshold=0.75  # <--- CRITICAL FIX
        )
    else:
        # Gemma Calibration (Default)
        weaver = UnitaryWeaver(
            model, tokenizer,
            dh_threshold=0.15,
            coherence_threshold=0.45
        )
    # -------------------

    print("\n=== EXPERIMENT 1: PHASE SPACE TOPOLOGY ===")
    # ... (rest of the code)
    print(f"{'Probe':<20} | {'State':<10} | {'Energy (E)':<10} | {'Structure (S)':<10}")
    print("-" * 60)

    for name, prompt in PROBES.items():
        inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        diag = weaver.topologist.diagnose(inputs)
        print(
            f"{name:<20} | {diag.state.value:<10} | {diag.observable.peak_dh:.4f}     | {diag.observable.concentration:.4f}")

    print("\n=== EXPERIMENT 2: THE FEW-SHOT TRAP ===")
    print(f"Prompt:\n{FEW_SHOT_TRAP.strip()}")
    inputs = tokenizer(FEW_SHOT_TRAP, return_tensors="pt").input_ids.to(model.device)

    # Diagnose
    diag = weaver.topologist.diagnose(inputs)
    print(f"\nDiagnosis: {diag.state.value} (S={diag.observable.concentration:.4f})")
    print(f"Strategy:  Alpha={diag.alpha}, Veto_K={diag.veto_k}")

    # Intervene
    output = weaver.evolve(inputs, max_tokens=10)
    print(f"Result:    {output}")

    if "10" in output:
        print("✅ SUCCESS: Truth Tunneled.")
    else:
        print("❌ FAILURE: Context Collapse.")


if __name__ == "__main__":
    run_collider()