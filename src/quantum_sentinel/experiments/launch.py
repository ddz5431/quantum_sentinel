import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from quantum_sentinel.sentinel.chronicle import SentinelAligner
from quantum_sentinel.engine.weaver import BacktrackingWeaver, QuantumLogitProcessor

logging.basicConfig(level=logging.INFO, format='%(message)s')


def run_alignment_session():
    model_id = "microsoft/Phi-3-mini-4k-instruct"

    print(f"✨ Initializing Quantum Sentinel on 3080 Ti...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Transitive Inference Task
    prompt = "A=B, B=C. Therefore, A="
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Framework Setup
    sentinel = SentinelAligner(tokenizer)
    processor = QuantumLogitProcessor(temperature=0.1)  # Low temp for sharp logic
    weaver = BacktrackingWeaver(model, tokenizer, sentinel, processor)

    print("🚀 Weaving Aligned Thread...")
    res = weaver.generate_with_alignment(
        inputs.input_ids,
        entropy_threshold=0.45,  # Recalibrated normalized threshold
        max_tokens=40
    )

    # Result Delivery
    final_text = tokenizer.decode(res['sequences'][0], skip_special_tokens=True)
    print("\n" + "=" * 50)
    print(f"EXPERIMENT RESULT: {final_text}")
    print(f"TOTAL REJECTIONS:  {res['backtracks']}")
    print("=" * 50)

    # Export scientific log
    sentinel.export_research_log()


if __name__ == "__main__":
    run_alignment_session()