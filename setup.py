from setuptools import setup, find_packages

setup(
    name="quantum_sentinel",
    version="0.1.0",
    description="Surgical Alignment and Temporal Coherence for Large Language Models",
    # description="A Centenary Framework for LLM Alignment via Quantum Backtracking",
    author="Sentinel Architect",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch",
        "transformers",
        "numpy",
        "matplotlib",
        "accelerate",
        "bitsandbytes"
    ],
    python_requires=">=3.9",
)