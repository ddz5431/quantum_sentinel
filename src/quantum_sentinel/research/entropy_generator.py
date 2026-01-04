import json
import random
from quantum_sentinel.paths import DATA_DIR


def generate_entropy_suite():
    """
    Shannon's Information Source: High-redundancy logical stress suite.
    Intentional structural repetition for statistical validity (N≥10 per category).
    """
    suite = []

    # Category definitions: (name, count, generator_function)
    categories = [
        ("Recency", 15, _generate_recency),
        ("Syllogism", 15, _generate_syllogism),
        ("Constraint", 10, _generate_constraint),
        ("Modus", 10, _generate_modus),
    ]

    for category, count, generator in categories:
        for i in range(count):
            prompt, target = generator()
            suite.append({
                "id": f"{category}-{i}",
                "category": category,
                "prompt": prompt,
                "target": target
            })

    # Save
    output_path = DATA_DIR / "entropy_signal.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(suite, f, indent=4)

    print(f"📡 [SIGNAL ENCODED] {len(suite)} stressors saved to {output_path}")
    return suite


def _generate_recency():
    v1, v2 = random.sample(range(10, 99), 2)
    prompt = f"Fact 1: The value is {v1}. Fact 2: The value is now {v2}. Question: What was the ORIGINAL value? Answer:"
    return prompt, str(v1)


def _generate_syllogism():
    a, b = random.choice([
        ("philosophers", "mortal"),
        ("birds", "animals"),
        ("dogs", "mammals"),
    ])
    name = random.choice(["Socrates", "Plato", "Rex", "Tweety"])
    prompt = f"All {a} are {b}. {name} is {b}. Is {name} definitely a {a[:-1]}? Answer (Yes/No):"
    return prompt, "No"


def _generate_constraint():
    word, length = random.choice([("fire", 5), ("water", 6), ("earth", 4)])
    prompt = f"Write a sentence about '{word}' using exactly {length} words:"
    return prompt, word


def _generate_modus():
    prompt = "Rule: If it rains, the ground is wet. Fact: The ground is NOT wet. Is it raining? Answer (Yes/No):"
    return prompt, "No"


if __name__ == "__main__":
    generate_entropy_suite()