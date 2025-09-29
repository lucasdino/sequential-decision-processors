import os, json

from sequential_decision_processors.prompts.chat_to_prompt import TokenizerCounter
from sequential_decision_processors.data.textworld.cleaned.generation_util.sft_dataloaders import DatasetSource, load_weighted_by_samples, load_weighted_by_tokens, write_token_csv_and_stats


# Main args to adjust
MAX_SAMPLES = 100_000
MAX_TOKENS  = 1_000_000
SAMPLING_STRATEGY = "get_token_stats"                # "samples" | "tokens" | "get_token_stats"
TOKENIZER_VERSION = "qwen25"
OUTPUT_FOLDER = "sequential_decision_processors/data/textworld/"
DATA_FOLDER = "sequential_decision_processors/data/textworld/cleaned/train_data"
DATASET_CONFIG = [
    {
        "name": "magpie",
        "files": ["magpie_20k.jsonl"],
        "weight": 0.25
    },
    # {
    #     "name": "rejection_sampling",
    #     "files": ["TBU"],
    #     "weight": 0.25
    # },
    {
        "name": "best_move",
        "files": ["cooking_singlemove_150k.jsonl"],
        "weight": 0.25
    },
    {
        "name": "best_line",
        "files": ["cooking_multimove_60k.jsonl"],
        "weight": 0.5
    },
]


# ------------------------------ sampling ------------------------------------
sources = [
    DatasetSource(
        name=cfg["name"],
        file_paths=[f"{DATA_FOLDER}/{fname}" for fname in cfg["files"]],
        weight=cfg["weight"],
    )
    for cfg in DATASET_CONFIG
]

final_samples = None
if SAMPLING_STRATEGY == "samples":
    final_samples = load_weighted_by_samples(sources, MAX_SAMPLES)
elif SAMPLING_STRATEGY == "tokens":
    token_counter = TokenizerCounter(TOKENIZER_VERSION)
    final_samples = load_weighted_by_tokens(sources, MAX_TOKENS, token_counter)
elif SAMPLING_STRATEGY == "get_token_stats":
    csv_path = os.path.join(OUTPUT_FOLDER, "cleaned/token_stats/token_stats.csv")
    token_counter = TokenizerCounter(TOKENIZER_VERSION)
    write_token_csv_and_stats(sources, token_counter, csv_path)
else:
    raise ValueError("SAMPLING_STRATEGY must be 'samples' or 'tokens'")

# ------------------------------ write outputs -------------------------------
if final_samples:
    print(f"Built {len(final_samples)} examples using strategy='{SAMPLING_STRATEGY}'")
    dataset_filename = f"llamafactory_programmatic_{len(final_samples)}.json"
    with open(f"{OUTPUT_FOLDER}/{dataset_filename}", "w", encoding="utf-8") as f:
        json.dump(final_samples, f, ensure_ascii=False, indent=2)

    datasets = {
        "llmchess_programmatic": {
            "file_name": dataset_filename,
            "columns": {"system": "system", "prompt": "user", "response": "assistant"},
        }
    }
    with open(f"{OUTPUT_FOLDER}/dataset_info.json", "w") as json_file:
        json.dump(datasets, json_file, indent=2)

    print(f"Wrote {len(final_samples)} rows → {OUTPUT_FOLDER}/{dataset_filename}")
    print(f"Dataset info saved to {OUTPUT_FOLDER}/dataset_info.json")