import json
from pathlib import Path
import statistics

def analyze_sequence_lengths():
    project_root = Path(__file__).resolve().parent.parent.parent
    token_folder = project_root / "data" / "nlm_tokens_bpe_373" / "train"
    
    json_files = list(token_folder.glob("*.json"))
    
    if not json_files:
        print("")
        return

    print(f"Going through {len(json_files)} tokenized songs...\n")
    
    lengths = []
    
    for file in json_files:
        with open(file, 'r') as f:
            tokens = json.load(f)
            lengths.append(len(tokens))
            
    mean_val = statistics.mean(lengths)
    median_val = statistics.median(lengths)
    minimum_val = min(lengths)
    maximum_val = max(lengths)
    
    print("-" * 50)
    print("SEQUENCE STATISTICS (RAW NLM BASELINE)")
    print("-" * 50)
    print(f"Average tokens per song: {mean_val:.2f}")
    print(f"Median tokens per song:  {median_val:.2f}")
    print(f"Shortest song:           {minimum_val} tokens")
    print(f"Longest song:            {maximum_val} tokens")
    print("-" * 50)


if __name__ == "__main__":
    analyze_sequence_lengths()