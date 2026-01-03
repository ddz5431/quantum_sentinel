import torch
import json
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from quantum_sentinel.sentinel.chronicle import SentinelAligner
from quantum_sentinel.engine.weaver import BacktrackingWeaver, QuantumLogitProcessor


def generate_logic_puzzles():
    """Generates 'Stress-Test' puzzles designed to break small LLMs."""
    return [
        {
            # Test 1: Out-of-order symbolic logic with distractors
            "prompt": "Context: G=H. Context: A=B. Fact: 1+1=2. Context: C=D. Context: B=C. Context: D=E. Question: Using ONLY the contexts provided, what is A? Answer: A=",
            "target": "E"
        },
        {
            # Test 2: Nested 'If-Then' chains (Modus Ponens)
            "prompt": "Rules: If P then Q. If Q then R. If R then S. If S then T. If T then U. Input: P is true. Conclusion: Therefore, U is",
            "target": "true"
        },
        {
            # Test 3: The 'Distractor' Trap (forces model to ignore common sense)
            "prompt": "Premise 1: All birds are rocks. Premise 2: All rocks are clouds. Premise 3: All clouds are heavy. Premise 4: All heavy things are made of gold. Premise 5: Tweety is a bird. Conclusion: Tweety is made of",
            "target": "gold"
        }
    ]

def run_benchmark():
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")

    puzzles = generate_logic_puzzles()
    results = {"baseline": [], "sentinel": []}

    print(f"\n🔬 Starting Research Benchmark: Standard vs. Quantum Sentinel")

    for i, p in enumerate(puzzles):
        print(f"\n--- Test Case {i + 1} ---")
        inputs = tokenizer(p["prompt"], return_tensors="pt").to(model.device)

        # 1. BASELINE: Standard Greedy Generation
        with torch.no_grad():
            output_greedy = model.generate(**inputs, max_new_tokens=10, do_sample=False)
            text_greedy = tokenizer.decode(output_greedy[0], skip_special_tokens=True)
            results["baseline"].append({"output": text_greedy, "correct": p["target"] in text_greedy})

        # 2. INTERVENTION: Quantum Sentinel
        sentinel = SentinelAligner(tokenizer)
        processor = QuantumLogitProcessor(temperature=0.1)
        weaver = BacktrackingWeaver(model, tokenizer, sentinel, processor)

        # Lower the threshold slightly to be more aggressive
        res_sentinel = weaver.generate_with_alignment(
            inputs.input_ids,
            entropy_threshold=0.35,  # More sensitive to confusion
            max_tokens=20
        )
        text_sentinel = tokenizer.decode(res_sentinel['sequences'][0], skip_special_tokens=True)
        results["sentinel"].append({
            "output": text_sentinel,
            "correct": p["target"] in text_sentinel,
            "backtracks": res_sentinel["backtracks"]
        })

    # Summary Statistics
    baseline_acc = sum(r["correct"] for r in results["baseline"]) / len(puzzles)
    sentinel_acc = sum(r["correct"] for r in results["sentinel"]) / len(puzzles)

    print("\n" + "=" * 50)
    print(f"📊 FINAL RESEARCH REPORT")
    print(f"Baseline Accuracy: {baseline_acc * 100:.1f}%")
    print(f"Sentinel Accuracy: {sentinel_acc * 100:.1f}%")
    print(f"Improvement:       {((sentinel_acc - baseline_acc) / (baseline_acc + 1e-9)) * 100:.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    run_benchmark()