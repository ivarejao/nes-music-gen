import json
import pickle
from pathlib import Path
from argparse import Namespace, ArgumentParser
from tqdm import tqdm

from src.nlm.tokenizer import NLMTokenizer


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--data-dir', required=True,
                        help='Root data directory containing train/valid/test splits with .nlm.pkl files')
    parser.add_argument('--tokenizer-path', required=True,
                        help='Path to the BPE tokenizer .json config file')
    parser.add_argument('--output-dir', default='out/',
                        help='Root output directory')
    parser.add_argument('--tokenize', '-t', action='store_true',
                        help='Apply tokenization (.nlm.pkl → .json token ids)')
    parser.add_argument('--detokenize', '-dt', action='store_true',
                        help='Apply detokenization (.json token ids → .nlm.pkl)')
    return parser.parse_args()


def tokenize_dataset(data_dir: Path, nlm_tokenizer: NLMTokenizer, output_dir: Path):
    print("Starting tokenization")

    output_dir = output_dir / "nlm_tokens"
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "valid", "test"]

    for split in splits:
        input_folder  = data_dir   / split
        output_folder = output_dir / split
        output_folder.mkdir(parents=True, exist_ok=True)

        pkl_files = list(input_folder.glob("*.nlm.pkl"))

        if not pkl_files:
            print(f"Warning: No .nlm.pkl files found in {input_folder}")
            continue

        print(f"\nProcessing {len(pkl_files)} '{split}' files...")

        for pkl_file in tqdm(pkl_files):
            try:
                tokens = nlm_tokenizer.tokenize(pkl_file)
                ids = tokens.ids

                clean_name = pkl_file.name.replace(".nlm.pkl", "")
                save_path  = output_folder / f"{clean_name}.json"

                with open(save_path, "w") as f:
                    json.dump(ids, f)

            except Exception as e:
                print(f"Error processing {pkl_file.name}: {e}")

    print("\nTokenization completed successfully!")


def detokenize_dataset(data_dir: Path, nlm_tokenizer: NLMTokenizer, output_dir: Path):
    print("Starting *de*-tokenization")

    output_dir = output_dir / "nlm_decoded"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(data_dir.glob("*.json"))

    if not json_files:
        print(f"No .json files found in {data_dir}.")
        return

    print(f"Found {len(json_files)} token files to detokenize...")

    for json_file in tqdm(json_files):
        try:
            with open(json_file, "r") as f:
                numeric_ids = json.load(f)

            print(f"\nDetokenizing: {json_file.name}  ({len(numeric_ids)} tokens)")

            # Decode ids back to the raw NLM text (e.g. "noteon_60 wait_10 ...")
            nlm_text = nlm_tokenizer.detokenize(numeric_ids)

            # Reconstruct the list of (function_name, value) tuples
            original_tuples = []
            for cmd in nlm_text.split():
                parts = cmd.rsplit("_", 1)
                if len(parts) == 2:
                    try:
                        function_name = parts[0]
                        value         = int(parts[1])
                        original_tuples.append((function_name, value))
                    except ValueError:
                        pass  # skip malformed tokens

            # Save reconstructed data as .nlm.pkl
            original_name = json_file.name.replace(".json", ".nlm.pkl")
            save_path     = output_dir / original_name

            with open(save_path, "wb") as f:
                pickle.dump(original_tuples, f, protocol=2)

            print(f"Saved → {save_path}")

        except Exception as e:
            print(f"Error while detokenizing {json_file.name}: {e}")

    print("\n" + "-" * 50)
    print("Detokenization completed successfully!")
    print(f"All .nlm.pkl files reconstructed in: {output_dir}")


def main():
    args = parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    nlm_tokenizer = NLMTokenizer(
        tokenizer_path=args.tokenizer_path,
        log_file=str(Path(args.output_dir) / "log.txt"),
    )

    print(f"Loaded tokenizer from {args.tokenizer_path}")
    print(f"Vocabulary size: {len(nlm_tokenizer.model.get_vocab())} tokens")

    if args.tokenize:
        tokenize_dataset(Path(args.data_dir), nlm_tokenizer, Path(args.output_dir))

    if args.detokenize:
        # Mirror midi_tokenize_dataset: detokenize reads from the test split of the token output
        detokenize_dataset(
            Path(args.output_dir) / "nlm_tokens" / "test",
            nlm_tokenizer,
            Path(args.output_dir),
        )

if __name__ == "__main__":
    main()