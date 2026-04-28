from datasets import load_dataset

dataset = load_dataset(
    "HuggingFaceFW/fineweb",
    name="sample-10BT",
    split="train",
    cache_dir="moe-llm/data",   # relative to wherever you run the script
)