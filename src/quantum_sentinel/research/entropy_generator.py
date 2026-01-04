import json
import random
from quantum_sentinel.paths import DATA_DIR


def generate_entropy_suite():
    """
    Generates a high-redundancy logical stress suite.
    """
    suite = []

    # 1. Recency-Bias
    for i in range(15):
        val1, val2 = random.sample(range(1, 100), 2)
        suite.append({
            "id": f"Recency-{i}",
            "category": "Recency",
            "prompt": f"Fact 1: The key is {val1}. Fact 2: The key has not changed. Fact 3: Actually, the key is {val2}. Question: What was the key originally? Answer:",
            "target": str(val1)
        })

    # 2. Syllogistic-Lures
    for i in range(15):
        suite.append({
            "id": f"Syllogism-{i}",
            "category": "Syllogism",
            "prompt": "All philosophers are mortal. Socrates is mortal. Therefore, Socrates is a",
            "target": "philosopher"
        })

    # 3. Constraint-Assay
    for i in range(10):
        target_len = random.randint(5, 8)
        suite.append({
            "id": f"Constraint-{i}",
            "category": "Constraint",
            "prompt": f"Instruction: Write a sentence about a cat using exactly {target_len} words. Result:",
            "target": "cat"
        })

    # 4. Modus-Tollens
    for i in range(10):
        suite.append({
            "id": f"Modus-{i}",
            "category": "Modus",
            "prompt": "Rule: If it is a square, it has four sides. Fact: This shape does not have four sides. Conclusion: This shape is",
            "target": "not a square"
        })

    # Save to correct location
    output_path = DATA_DIR / "entropy_signal.json"  # ← FIXED: Correct path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(suite, f, indent=4)

    print(f"📡 [SIGNAL ENCODED] {len(suite)} stressors saved to {output_path}")
    return suite


if __name__ == "__main__":
    generate_entropy_suite()