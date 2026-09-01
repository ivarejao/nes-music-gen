import json
import os 
import pandas as pd
from argparse import ArgumentParser, Namespace
from pathlib import Path
from tqdm import tqdm

from src.midi.MIDI import MIDIHandler
from src.nlm.NLM import NLMHandler

def parse_args() -> Namespace:
    parser = ArgumentParser(description='Post processing after main augmentation script is done')
    parser.add_argument('--dataset-path', type=str, default='dataset/complete/')
    parser.add_argument('--midi-tokenizer', help='Define midi-tokenizer', default='models/tokenizers/symbolic/midi_mmm_tokenizer.json')
    parser.add_argument('--nlm-tokenizer', help='Define nlm-tokenizer', default='models/tokenizers/symbolic/nlm_bpe_373_tokenizer.json')

    args = parser.parse_args()
    return args



# 1. Walk through the dataset, only walk on the new subsamples
# 2. For each new sample:
# 2.a. Tokenize the midi and NLM with the name {track_file_name}_{format}_tokens.json
# 2.b. Tag the new samples
# 3. Join the new metadata


def add_tags(dataset_metadata : pd.DataFrame, dataset_path : Path):

    has_tags_mask = dataset_metadata.has_midi & (dataset_metadata.duration_s <= 60)
    target_samples = dataset_metadata[has_tags_mask]

    without_tags = dataset_metadata[~has_tags_mask]
    without_tags[["has_tags", "file_path_tags"]] = None
    without_tags["console"] = "NES"
    without_tags[[
        "tempo_bpm", "tempo_caption", "key", "time_signature", "duration_caption", "instruments",
        "chord_summary", "chords"
    ]] = None

    print("Getting tags from target samples")
    with_tags = []
    for (i, sample) in tqdm(target_samples.iterrows(), desc="Loading tags"):
        sample["has_music_track_description"] = True
        sample["file_path_music_track_description"] = \
            Path(sample["id"]) / sample["track_name"] / \
            (sample["file_name_prefix"] + "_music_track_description.md")
        

        # Also update the midi and nlm tokens
        sample["file_path_midi_tokens"] = \
            Path(sample["id"]) / sample["track_name"] / \
            (sample["file_name_prefix"] + "_midi_tokens.json")
        sample["file_path_nlm_tokens"] = \
            Path(sample["id"]) / sample["track_name"] / \
            (sample["file_name_prefix"] + "_nlm_tokens.json")

        # load tags
        sample["has_tags"] = False
        sample["file_path_tags"] = \
            Path(sample["id"]) / sample["track_name"] / \
            (sample["file_name_prefix"] + "_tags.json")

        tags_path = Path(dataset_path) / sample["file_path_tags"]
        with open(tags_path, "r") as f:
            tags = json.load(f)

        sample["console"] = tags["console"]
        sample["tempo_bpm"] = tags["tempo"][0]
        sample["tempo_caption"] = tags["tempo"][-1]
        sample["key"] = tags["key"]
        sample["time_signature"] = tags["time_signature"]
        sample["duration_s"] = tags["duration"][0]
        sample["duration_caption"] = tags["duration"][-1]
        sample["instruments"] = tags["instruments"]
        sample["chord_summary"] = tags["chord_summary"]
        sample["chords"] = tags["chords"]

        sample = sample.reindex(index=without_tags.columns)

        with_tags.append(sample)

        # TODO: remove .csv if exists and remove .tags
        # One time only
        csv_file = Path(sample["id"]) / sample["track_name"] / \
                    (sample["file_name_prefix"] + ".csv")

        if csv_file.exists():
            csv_file.unlink()
        print(sample["file_name_prefix"])

    merged_df = pd.concat([without_tags, pd.DataFrame(with_tags)], ignore_index=True, sort=False)
    merged_df = merged_df.sort_values(by="file_name_prefix", kind="mergesort").reset_index(drop=True)
    merged_df.to_csv(dataset_path / "nes_metadata.csv", index=False)

    # add has_tags, file_path_tags + tags


def post_processing_dataset(dataset_metadata: pd.DataFrame, dataset_path : Path, nlm_tokenizer_path : Path, midi_tokenizer_path : Path):
    

    dataset_path = Path(dataset_path)
    nlm_tokenizer_path = Path(nlm_tokenizer_path)
    midi_tokenizer_path = Path(midi_tokenizer_path)

    nlm_handler = NLMHandler(tokenizer_path=nlm_tokenizer_path)
    midi_handler = MIDIHandler(tokenizer_path=midi_tokenizer_path)


    new_samples = dataset_metadata[dataset_metadata.file_path_midi_tokens.isna() & dataset_metadata.file_path_nlm_tokens.isna()]

    for (_, sample) in tqdm(new_samples.iterrows(), desc="Tokenizing "):

        print("{}".format(sample["file_name_prefix"]))
        # MIDI
        midi_path = sample["file_path_midi"]
        midi_path = dataset_path / midi_path
        midi_tokens_path = midi_path.with_name(midi_path.name.replace(".mid", "_midi_tokens.json") )

        midi_tokens = midi_handler.tokenize(midi_path)
        midi_ids = midi_tokens[0].ids if isinstance(midi_tokens, list) else midi_tokens.ids
        with open(midi_tokens_path, "w") as f:
            json.dump(midi_ids, f)

        # NLM
        nlm_path = sample["file_path_nlm"]
        nlm_path = dataset_path / nlm_path
        nlm_tokens_path = nlm_path.with_name(nlm_path.name.replace(".nlm.pkl", "_nlm_tokens.json"))

        nlm_tokens = nlm_handler.tokenize(nlm_path)
        nlm_ids = nlm_tokens.ids
        with open(nlm_tokens_path, "w") as f:
            json.dump(nlm_ids, f)

def main():
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    nlm_tokenizer_path = Path(args.nlm_tokenizer)
    midi_tokenizer_path = Path(args.midi_tokenizer)

    df = pd.read_csv(dataset_path / "nes_metadata.csv", dtype={"id": str})
    # post_processing_dataset(df, dataset_path, nlm_tokenizer_path, midi_tokenizer_path)
    add_tags(df, dataset_path)

if __name__ == "__main__":
    main()