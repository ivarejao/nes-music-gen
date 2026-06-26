import json
import pickle
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


def reconstruct_complete_dataset():
    project_root = Path("/home/igor/Desktop/Repos/nes-music-gen")
    base_tokens_folder = project_root / "data" / "nlm_tokens_bpe_373"
    

    base_output_folder = project_root / "data" / "generated_nlm_v1"
    bpe_path = project_root / "models" / "tokenizers" / "lmf_bpe_373_tokenizer.json"


    tokenizer = Tokenizer.from_file(str(bpe_path))
    tokenizer.decoder = ByteLevelDecoder()

    splits = ["train", "valid", "test"]

    for split in splits:
        input_folder = base_tokens_folder / split
        output_folder = base_output_folder / split
        
        
        output_folder.mkdir(parents=True, exist_ok=True)

        json_files = list(input_folder.glob("*.json"))
        if not json_files:
            continue

        print(f"Reconstructing {len(json_files)} songs from the '{split}' split...")

        for json_file in json_files:
            with open(json_file, 'r') as f:
                compressed_ids = json.load(f)

            # 1. Transforms the number matrix into the long string
            recovered_text = tokenizer.decode(compressed_ids)

            # 2. Breaks the string into isolated commands
            text_commands = recovered_text.split()

            # 3. Reconstructs the list of tuples required by the emulator
            original_tuples = []
            for cmd in text_commands:
                parts = cmd.rsplit('_', 1)
                
                if len(parts) == 2:
                    try:
                        function_name = parts[0]
                        value = int(parts[1])
                        original_tuples.append((function_name, value))
                    except ValueError:
                        pass

            # 4. Saves the list back in Python Pickle (.pkl) format
            original_name = json_file.name.replace(".json", ".nlm.pkl")
            save_path = output_folder / original_name

            with open(save_path, 'wb') as f_out:
                # The dump freezes the variable from RAM straight to the HDD
                pickle.dump(original_tuples, f_out)

    print("-" * 50)
    print("Reverse engineering completed successfully!")
    print(f"All machine files (.pkl) were reconstructed in the folder: {base_output_folder}")


if __name__ == "__main__":
    reconstruct_complete_dataset()