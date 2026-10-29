import os
import shutil
import pandas as pd

DATASET_ROOT = "/home/es119256/dados/datasets/nes_dataset"
FOLDER_PATH = "/home/es119256/dados/repos/nes-music-gen/dataset/test_pre_processed"

def copy_tokenized_data(dataset_root: str = DATASET_ROOT, dest_folder: str = FOLDER_PATH):
    csv_path = os.path.join(dataset_root, "nes_metadata.csv")

    os.makedirs(dest_folder, exist_ok=True)

    # Get clean df
    df = pd.read_csv(csv_path, dtype={"id":"object"})
    df = df[df.has_midi & df.has_nlm & df.has_music_track_description & (df.split == "test")]
    df:pd.DataFrame = df.reset_index()

    for row in df.to_dict(orient='records'):
        file_name_prefix = row["file_name_prefix"]
        wav_path = os.path.join(dataset_root, row["file_path_wav"])

        origin_path = wav_path[:-4] + '_pre_comp.pt'

        dest_filename = f"{file_name_prefix}_pre_comp.pt"
        dest = os.path.join(dest_folder, dest_filename)

        if os.path.exists(origin_path):
            shutil.copy(origin_path, dest)
            print(f"Copied {file_name_prefix}")
        else:
            print(f"File not found: {origin_path}")
        print(f"Copied {file_name_prefix}")

if __name__ == '__main__':
    copy_tokenized_data()