import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from quantum_sentinel.sentinel.chronicle import SentinelAligner
from quantum_sentinel.engine.weaver import BacktrackingWeaver, QuantumLogitProcessor

# Configure logging to observe real-time Sentinel interventions
logging.basicConfig(level=logging.INFO, format='%(message)s')

def run_quantum_sentinel_benchmark():
    """
    Experimental suite proving that Entropy Velocity (ΔH) and
    Ground State Injection resolve deterministic contradictions.
    """
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load in bfloat16 for stability on 3080 Ti hardware
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Scenarios designed to trigger high-entropy 'Phase Transitions'
    recovery_tests = [
        {
            "id": "Recency-Bias-Conflict",
            "prompt": "Fact 1: X=5. Fact 2: Y=X. Fact 3: X=10. Question: What is the value of Y? Answer: Y=",
            "target": "5"
        },
        {
            "id": "Logical-Modus-Tollens",
            "prompt": "Fact 1: If it is a square, it has four sides. Fact 2: This shape does not have four sides. Therefore, the shape is",
            "target": "not a square"
        },
        {
            "id": "Deterministic-Contradiction",
            "prompt": "Fact 1: All birds fly. Fact 2: Penguins are birds. Fact 3: Penguins do not fly. Conclusion: The statement 'All birds fly' is",
            "target": "false"
        }
    ]

    for test in recovery_tests:
        print(f"\n" + "=" * 60)
        print(f"🚀 [TRIAL: {test['id']}]")
        print(f"=" * 60)

        inputs = tokenizer(test["prompt"], return_tensors="pt").to(model.device)

        # 1. RUN BASELINE (Greedy Evolution)
        # Represents the unobserved deterministic path susceptible to bias.
        with torch.no_grad():
            out_greedy = model.generate(**inputs, max_new_tokens=15, do_sample=False)
            text_greedy = tokenizer.decode(
                out_greedy[0][inputs.input_ids.shape[-1]:],
                skip_special_tokens=True
            ).strip()

        # 2. RUN SENTINEL (Injection & Phase Transition Monitoring)
        sentinel = SentinelAligner(tokenizer, task_type=test['id'])
        processor = QuantumLogitProcessor(temperature=0.1, repetition_penalty=1.5)
        weaver = BacktrackingWeaver(model, tokenizer, sentinel, processor)

        # Generate with alignment, monitoring the 'Injection Delta'
        res = weaver.generate_with_alignment(
            inputs.input_ids,
            kl_threshold=15.0, # High KL allows for semantic divergence during reasoning
            max_tokens=15
        )

        text_sentinel = tokenizer.decode(
            res['sequences'][0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        ).strip()

        # 3. QUANTITATIVE ANALYSIS
        # Extract Phase Transitions and Injection Surges
        injection_events = [p for p in sentinel.phase_log if p.get("is_injection", False)]

        print(f"   Baseline Output:  {text_greedy}")
        print(f"   Sentinel Output:  {text_sentinel}")
        print(f"   Backtracks:       {res['backtracks']}")

        if injection_events:
            avg_delta = sum([e['injection_surge'] for e in injection_events]) / len(injection_events)
            print(f"   ✨ Injection Delta: +{avg_delta:.4f} Prob Surge")

        success = test['target'].lower() in text_sentinel.lower()
        print(f"   Final Status:     {'✅ SUCCESS' if success else '❌ DEADLOCK'}")

        # Export research-grade logs for paper results
        sentinel.print_multiverse_report()
        sentinel.export_research_log(path=f"results/alignment_{test['id']}.json")

if __name__ == "__main__":
    run_quantum_sentinel_benchmark()