import json
import random
from pathlib import Path


# noinspection PyDuplication
def generate_entropy_suite():
    """
    Shannon's Information Source: Generates the 50-stressor suite.
    Uses a 'Unitary Template' to minimize code noise (IDE duplications).
    """
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    output_path = project_root / "data" / "entropy_signal.json"

    # Define the 'Signal Archetypes'
    archetypes = {
        "Recency": "Initial: X={v1}. Update: X={v2}. Question: What was X before? Answer: X=",
        "Syllogism": "Premise: All A are B. Fact: X is B. Question: Is X definitely A? Answer:",
        "Constraint": "Instruction: Write a sentence about '{word}' with exactly {val} words. Result:",
        "Modus": "Rule: If P, then Q. Fact: Not Q. Conclusion: Therefore, Not"
    }

    suite = []

    # Generate 50 signals across the archetypes
    for i in range(50):
        if i < 15:  # Recency
            v1, v2 = random.sample(range(10, 99), 2)
            prompt = archetypes["Recency"].format(v1=v1, v2=v2)
            target = str(v1)
            cat = "Recency"
        elif i < 30:  # Syllogism
            prompt = archetypes["Syllogism"]
            target = "No"
            cat = "Syllogism"
        elif i < 40:  # Constraint
            word, val = random.choice([("fire", 5), ("water", 6), ("earth", 4)])
            prompt = archetypes["Constraint"].format(word=word, val=val)
            target = word
            cat = "Constraint"
        else:  # Modus
            prompt = archetypes["Modus"]
            target = "P"
            cat = "Modus"

        suite.append({"id": f"{cat}-{i}", "prompt": prompt, "target": target})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(suite, f, indent=4)

    print(f"📡 [SIGNAL ENCODED] Universal Suite archived at: {output_path}")


if __name__ == "__main__":
    generate_entropy_suite()