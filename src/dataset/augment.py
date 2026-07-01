#
# Clean MIDI data to:
# 1) Remove all pieces that are not in common time (i.e., 4/4)
# 2) Remove all non-piano tracks
# 3) Combine multiple piano tracks into a single one
#
# Lucas N. Ferreira
# lucasnfe@gmail.com
#
#
# TODO: Organize

import os
import copy
import argparse
import numpy as np
import multiprocessing as mp
import itertools
import pandas as pd
from urllib.parse import urljoin
import requests


from pathlib import Path

import miditoolkit
from miditoolkit.midi.containers import Instrument, TempoChange, Note
from tqdm import tqdm

# Range of transposition intervals (in semi-tones)
transpose_intervals = range(-4, 5)

# Range of strech factors
stretch_factors = [0.8, 0.9, 1.0]

def transpose(midi_obj, interval):
    # Transpose all pitched notes
    for instrument in midi_obj.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                note.pitch += interval

def strech(midi_obj, stretch_factor):
    for tempo_change in midi_obj.tempo_changes:
        tempo_change.tempo = round(tempo_change.tempo * stretch_factor)


def convert(input_midi_path : str, output_dir : str, root_dir : str, convertion_key : str):

    API_URL = "http://localhost:5555"
    input_midi_path = Path(input_midi_path)
    output_dir = Path(output_dir)

    input_midi_path = input_midi_path.absolute()
    output_dir = output_dir.absolute()

    # convert files to the respectively docker path
    input_midi_path = Path("/data") / input_midi_path.relative_to(root_dir)
    output_dir = Path("/data") / output_dir.relative_to(root_dir)
    
    endpoint = urljoin(API_URL, "convert")
    params = {
        "convertionkey" : convertion_key,
        "path_to_input" : str(input_midi_path),
        "output_dir" : str(output_dir)
    }
    response = requests.post(endpoint, json=params)
    if response.status_code == 200:
        output_file_path = response.json()["file_path"]
        print("File {} sythesized onto file {}".format(os.path.basename(input_midi_path), output_file_path))
        return output_file_path
    else:
        print("Error on synthesis of file {}\n{}".format(input_midi_path, response.json()["error"]))
        return None



def proc_one(reference : pd.Series, root_dir : str):
    print('----')
    print(' >', reference["file_path_midi"])

    midi_obj = miditoolkit.midi.parser.MidiFile(
        Path(root_dir) / reference["file_path_midi"]
    )

    variations = list(itertools.product(transpose_intervals, stretch_factors))

    for (interval, factor) in variations:
        # create descriptions for each modification
        # tranpose
        if interval < 0:
            interval_suffix = "_b_" + str(abs(interval))
        elif interval == 0:
            interval_suffix = "_original"
        elif interval > 0:
            interval_suffix = "_#_" + str(abs(interval))

        # stretch
        if factor < 1.0:
            strech_suffix = "_slow_" + str(int(factor * 100))
        elif factor == 1.0:
            strech_suffix = "_original"
        elif factor > 1.0:
            strech_suffix = "_fast_" + str(int(factor * 100))

        sample_signature_name = interval_suffix + strech_suffix

        # create directory
        base_dir = Path(
            os.path.join(root_dir, str(reference["id"]), reference["track_name"] + sample_signature_name)
        )

        base_dir.mkdir(exist_ok=True)

        # Make a copy of the original midi
        transpose_mid = copy.deepcopy(midi_obj)

        # Transpose midi
        transpose(transpose_mid, interval)

        # Strech midi
        transposed_streched_mid = copy.deepcopy(transpose_mid)

        # Strech midi
        strech(transposed_streched_mid, factor)

        midi_new_path = base_dir / (reference["file_name_prefix"] + sample_signature_name + ".mid")

        # save midi
        print(' >', midi_new_path)
        transposed_streched_mid.dump(midi_new_path)

        # convert to nlm
        nlm_new_path = convert(midi_new_path, os.path.dirname(midi_new_path), root_dir, "midi_to_nlm")

        # convert to wav
        wav_new_path = convert(midi_new_path, os.path.dirname(midi_new_path), root_dir, "midi_to_wav")


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='augment.py')
    parser.add_argument('--dataset-path', type=str, required=True)
    args = parser.parse_args()

    root_dir = Path(args.dataset_path).absolute()

    df = pd.read_csv(root_dir / "nes_metadata.csv")
    df["id"] = df.id.map('{:03d}'.format)
    df = df[~(df.split == "test")]

    print(root_dir)

    # collect
    data = []
    for idx, sample in df.iterrows():
        # append
        data.append([sample, root_dir])
        if idx > 20:
            break

    # data = [[
    #     df[(df.id == "310") & (df.track_name == "Cantina")].iloc[0],
    #     root_dir
    # ]]
    # proc_one(*data[0])
    # run, multi-thread
    pool = mp.Pool()
    pool.starmap(proc_one, data)
