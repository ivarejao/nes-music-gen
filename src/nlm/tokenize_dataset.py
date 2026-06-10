import json
import pickle
from pathlib import Path
from tokenizers import Tokenizer


def tokenize_dataset_bpe():
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_nlm_folder = project_root / "data" / "nesmdb_nlm"

    path_tokens = project_root / "data" / "nlm_tokens_bpe_373"

    path_bpe = (
        project_root / "models" / "tokenizers" / "lmf_bpe_373_tokenizer.json"
    )

    tokenizer = Tokenizer.from_file(str(path_bpe))

    splits = ["train", "valid", "test"]

    for split in splits:
        input_folder = raw_nlm_folder / split
        output_folder = path_tokens / split
        output_folder.mkdir(parents=True, exist_ok=True)

        pkl_files = list(input_folder.glob("*.pkl"))
        if not pkl_files:
            continue

        print(f"Processing {len(pkl_files)} '{split}' files...")

        for pkl_file in pkl_files:
            with open(pkl_file, "rb") as f:
                nlm_data = pickle.load(f)

            text_line = " ".join([f"{cmd[0]}_{cmd[1]}" for cmd in nlm_data])

            encoded = tokenizer.encode(text_line)

            clean_name = pkl_file.name.replace(".nlm.pkl", "")
            save_path = output_folder / f"{clean_name}.json"

            with open(save_path, "w") as f:
                json.dump(encoded.ids, f)


if __name__ == "__main__":
    tokenize_dataset_bpe()