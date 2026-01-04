import torch
import pandas as pd
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

from quantum_sentinel.paths import DATA_DIR, ASSAYS_DIR
from quantum_sentinel.core.shannon_observer import ShannonObserver
from quantum_sentinel.core.unitary_weaver import UnitaryWeaver


def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=True
    )

    return model, tokenizer


def run_universal_assay(model_id):
    signal_path = DATA_DIR / "entropy_signal.json"

    with open(signal_path, "r") as f:
        suite = json.load(f)

    model, tokenizer = load_model(model_id)

    results = []
    for case in suite:
        observer = ShannonObserver(tokenizer, case['id'])
        # Ensure UnitaryWeaver is initialized with the observer
        weaver = UnitaryWeaver(model, tokenizer, observer)

        inputs = tokenizer(case['prompt'], return_tensors="pt").to(model.device)
        out = weaver.evolve(inputs.input_ids)

        # 📡 WAVEFUNCTION TRACE: Save the token-by-token entropy logs
        observer.save_trace(model_id.split('/')[-1])

        # 🔍 AUDIT LOGIC: Isolate the generated sequence from the prompt
        prompt_len = inputs.input_ids.shape[-1]
        generated_ids = out['sequences'][0][prompt_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Determine success based ONLY on the generated output
        target = case['target'].lower()
        actual = generated_text.lower()

        # We perform a "Containment Check" but store the raw output for the Audit Report
        is_successful = target in actual

        results.append({
            "model": model_id,
            "category": case['id'].split('-')[0],
            "peak_dh": max([p['velocity'] for p in observer.phase_log]) if observer.phase_log else 0.0,
            "success": is_successful,
            "output_raw": generated_text,  # Capture the raw signal
            "target_expected": case['target']  # Capture the expected ground state
        })
        torch.cuda.empty_cache()

    # Flush VRAM for the next architecture
    del model
    torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    ASSAYS_DIR.mkdir(parents=True, exist_ok=True)

    # The 2026 Triple-Architecture Cross-Validation Set
    models = [
        "microsoft/Phi-3-mini-4k-instruct",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "google/gemma-3-1b-it"
    ]

    final_data = []
    for m in models:
        print(f"🔬 Running audited assay on {m}...")
        try:
            final_data.extend(run_universal_assay(m))
        except Exception as e:
            print(f"❌ Error during assay of {m}: {e}")

    df = pd.DataFrame(final_data)

    # Save the Audited Dataset
    output_path = ASSAYS_DIR / "final_decoherence_2026.csv"
    df.to_csv(output_path, index=False)

    print(f"\n📊 [AUDITED ASSAY COMPLETE]")
    print(f"Results archived in: {output_path}")

    # Summary of 'Violent' vs 'Silent' decoherence signatures
    summary = df.groupby('category').agg({
        'peak_dh': 'mean',
        'success': 'mean'
    })
    print("\nPre-Audit Summary Statistics:")
    print(summary)
