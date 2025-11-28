import os, json
from sequential_decision_processors.prompts.chat_to_prompt import TokenizerCounter
from sequential_decision_processors.data.generation_util.sft_dataloaders import DatasetSource, load_weighted_by_samples, load_weighted_by_tokens, write_token_csv_and_stats


# Main args to adjust
MAX_SAMPLES = 750_000
MAX_TOKENS  = 4_000_000
SAMPLING_STRATEGY = "tokens"                # "samples" | "tokens" | "get_token_stats"
TOKENIZER_VERSION = "qwen25"
OUTPUT_FOLDER = "sequential_decision_processors/data"
DATA_FOLDER = "sequential_decision_processors/data/cleaned_sft"
DATASET_CONFIG = [
    {
        "name": "mgp",
        "files": ["magpieclean_20k.jsonl"],
        "weight": 0.465
    },
    {
        "name": "rft",
        "files": ["rft_alfworld_906.jsonl", "rft_combined_3327.jsonl", "rft_twx_1178.jsonl"],
        "weight": 0.465
    },
    # {
    #     "name": "bestmove",
    #     "files": ["bestmove_30k.jsonl", "bestmove_30k_copy.jsonl"],
    #     "weight": 0.05
    # },
    {
        "name": "bestline",
        "files": ["bestline_4k.jsonl"],
        "weight": 0.07
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
    csv_path = os.path.join(OUTPUT_FOLDER, "token_stats.csv")
    token_counter = TokenizerCounter(TOKENIZER_VERSION)
    write_token_csv_and_stats(sources, token_counter, csv_path)
else:
    raise ValueError("SAMPLING_STRATEGY must be 'samples' or 'tokens'")

# ------------------------------ write outputs -------------------------------
if final_samples:
    print(f"Built {len(final_samples)} examples using strategy='{SAMPLING_STRATEGY}'")
    dataset_filename = f"llamafactory_sdp_sftdata.json"
    with open(f"{OUTPUT_FOLDER}/{dataset_filename}", "w", encoding="utf-8") as f:
        json.dump(final_samples, f, ensure_ascii=False, indent=2)

    datasets = {
        "sdp_sft_data": {
            "file_name": dataset_filename,
            "columns": {"system": "system", "prompt": "user", "response": "assistant"},
        }
    }
    with open(f"{OUTPUT_FOLDER}/dataset_info.json", "w") as json_file:
        json.dump(datasets, json_file, indent=2)

    print(f"Wrote {len(final_samples)} rows → {OUTPUT_FOLDER}/{dataset_filename}")
    print(f"Dataset info saved to {OUTPUT_FOLDER}/dataset_info.json")