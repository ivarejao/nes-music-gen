from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
from time import time

import pandas as pd
from tqdm import tqdm


def _resolve_dataset_dir(dataset_dir: str | None) -> Path:
    if dataset_dir is not None:
        return Path(dataset_dir).expanduser().resolve()

    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / "dataset" / "complete").resolve()


def merge_complete_csvs(
    dataset_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    base_name: str = "nes_metadata.csv",
) -> Path:
    dataset_root = _resolve_dataset_dir(str(dataset_dir) if dataset_dir is not None else None)
    base_csv = dataset_root / base_name

    if not base_csv.exists():
        raise FileNotFoundError(f"Base CSV not found: {base_csv}")

    print("Loading base csv")
    base_df = pd.read_csv(base_csv)
    output_file = Path(output_path).expanduser().resolve() if output_path else dataset_root / "merged_complete.csv"

    print("Scanning csv files")
    pattern = re.compile(r".*(_b_1|_#_1|_original)(_slow_80|_fast_120|_original)\.csv$")

    # csv_files = [
    #     path for path in tqdm(dataset_root.rglob("*.csv"), desc="Listing files")
    #     if path.is_file() and pattern.search(path.name)
    # ]
    start = time()
    csv_files = list(filter(lambda path: path.is_file() and pattern.search(path.name), dataset_root.rglob("*.csv")))
    csv_files = list(map(str, csv_files))
    end = time()
    print(str(end - start))

    print("Starting glue of {} files".format(len(csv_files)))
    with open("new_csv_files.txt", "w") as f:
        f.write("\n".join(csv_files))
    exit()
    # remove, is not needede anymore
    base_df.remove(columns=["tempo_bpm", "tempo_conf", "key"], inplace=True)
    all_frames = [base_df]
    for csv_path in tqdm(csv_files, desc="Merging CSV files", unit="file"):
        df = pd.read_csv(csv_path)
        # Fix
        df["file_name_prefix"] = os.path.basename(df["file_path_midi"]).replace(".mid", "")
        df["track_name"] = os.path.dirname(df["file_path_midi"]).split("/")[0]
        df["duration_s"] = df["duration"]
        df["has_music_track_description"] = False
        df["file_path_vgm"] = None

        df.remove(columns=["tempo_bpm", "tempo_conf", "key"], inplace=True)
        df.remove(columns=["duration", "has_vgm"], inplace=True)

        if set(df.columns) != set(base_df.columns):
            missing = [col for col in base_df.columns if col not in df.columns]
            extra = [col for col in df.columns if col not in base_df.columns]
            raise ValueError(
                f"CSV schema mismatch in {csv_path}. "
                f"Missing columns: {missing or 'none'}; extra columns: {extra or 'none'}"
            )

        # Preserve the canonical column order from the base CSV.
        df = df.reindex(columns=base_df.columns)
        all_frames.append(df)

    merged_df = pd.concat(all_frames, ignore_index=True, sort=False)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_file, index=False)

    print(f"Merged {len(csv_files) + 1} CSV files into {output_file}")
    print(f"Total rows: {len(merged_df)}")
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge all CSV files in dataset/complete using nes_metadata.csv as the canonical schema."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Directory containing the dataset CSV files. Defaults to dataset/complete under the repo root.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Destination CSV path for the merged file. Defaults to dataset/complete/merged_complete.csv.",
    )
    parser.add_argument(
        "--base-name",
        type=str,
        default="nes_metadata.csv",
        help="Canonical base CSV name to use as schema reference.",
    )
    args = parser.parse_args()

    merge_complete_csvs(dataset_dir=args.dataset_dir, output_path=args.output, base_name=args.base_name)


if __name__ == "__main__":
    main()
