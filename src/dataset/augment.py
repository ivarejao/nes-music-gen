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
import time
import wave

from pathlib import Path

import miditoolkit
from miditoolkit.midi.containers import Instrument, TempoChange, Note
from tqdm import tqdm

from src.utils import setup_logger


DESCRIPTOR_API_URL = "http://localhost:8000"
NESMDB_API_URL = "http://localhost:5555"


# Range of transposition intervals (in semi-tones)
transpose_intervals = range(-4, 5)

# Range of strech factors
stretch_factors = [0.8, 0.9, 1.0]

def _serializable(field):
    return [str(i) for i in field] if isinstance(field, list) else str(field)

def transpose(midi_obj, interval):
    # Transpose all pitched notes
    for instrument in midi_obj.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                note.pitch += interval

def strech(midi_obj, stretch_factor):
    for tempo_change in midi_obj.tempo_changes:
        tempo_change.tempo = round(tempo_change.tempo * stretch_factor)

def crop_wav(input_path, seconds=60, output_path=None):
    with wave.open(input_path, 'rb') as w:
        params = w.getparams()
        framerate = w.getframerate()
        n_frames = w.getnframes()
        duration = n_frames / framerate

        # Check the signal size first
        if duration < seconds:
            return None

        # Otherwise, do the cropping
        max_frames = int(seconds * framerate)
        data = w.readframes(max_frames)

    if output_path is None:
        output_path = input_path

    with wave.open(output_path, 'wb') as o:
        o.setparams(params)
        o.writeframes(data)
    print("Wav signal cropped on {}".format(output_path))


def get_wav_duration(file_path):
    with wave.open(str(file_path), 'rb') as audio_file:
        # Get the total number of audio frames
        frames = audio_file.getnframes()
        # Get the sampling frequency (frames per second)
        rate = audio_file.getframerate()
        # Calculate duration in seconds
        duration = frames / float(rate)
        return duration

def convert(input_midi_path : str, output_dir : str, root_dir : str, convertion_key : str):

    input_midi_path = Path(input_midi_path)
    output_dir = Path(output_dir)

    input_midi_path = input_midi_path.absolute()
    output_dir = output_dir.absolute()

    # convert files to the respectively docker path
    input_midi_path = Path("/data") / input_midi_path.relative_to(root_dir)
    output_dir = Path("/data") / output_dir.relative_to(root_dir)
    
    endpoint = urljoin(NESMDB_API_URL, "convert")
    params = {
        "convertionkey" : convertion_key,
        "path_to_input" : _serializable(input_midi_path),
        "output_dir" : _serializable(output_dir)
    }
    response = requests.post(endpoint, json=params)
    if response.status_code == 200:
        output_file_path = response.json()["file_path"]
        print("File {} sythesized onto file {}".format(os.path.basename(input_midi_path), output_file_path))
        return output_file_path
    else:
        print("Error on synthesis of file {}\n{}".format(input_midi_path, response.json()["error"]))
        return None

def call_madmom(endpoint: str, tracks: list[str], out_dir: str | None = None, timeout: int = 300) -> dict:
    """
    Call a madmom API endpoint ("tempo" or "key") with a list of track paths.

    Args:
        endpoint: "tempo" or "key"
        tracks:   absolute paths (inside the container) to the .wav files (max 10)
        out_dir:  optional output directory override
        timeout:  request timeout in seconds

    Returns:
        Parsed JSON response as a dict.
    """
    # just ensure the paths is strings in order to be seriazable
    payload = {"tracks": _serializable(tracks)}
    if out_dir is not None:
        payload["out_dir"] = _serializable(out_dir)

    resp = requests.post(f"{DESCRIPTOR_API_URL}/{endpoint}", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

# convenience wrappers
def detect_tempo(tracks, out_dir=None, **kw):
    return call_madmom("tempo", tracks, out_dir, **kw)

def recognize_key(tracks, out_dir=None, **kw):
    return call_madmom("key", tracks, out_dir, **kw)

def _basename(path: str) -> str:
    """Return the .wav filename from a full track path."""
    return os.path.basename(path)

def parse_tempo(content: str):
    """
    TempoDetector output: lines of `tempo<TAB>strength`, dominant tempo first.
    Returns (bpm, confidence) as floats.
    """
    if not content:
        return None, None
    first = content.strip().splitlines()[0].split()
    bpm = float(first[0])
    conf = float(first[1]) if len(first) > 1 else None
    return bpm, conf

def parse_key(content: str):
    """KeyRecognition output: a single key string, e.g. 'C major'."""
    return content.strip() if content else None

def apply_results(new_samples, tempo=None, key=None):
    """
    Update metadata Series in new_samples from API responses.
    Matches by .wav basename so it works regardless of the full path key.
    """
    if tempo:
        for track, content in tempo["results"].items():
            name = _basename(track)
            if name in new_samples:
                bpm, conf = parse_tempo(content)
                new_samples[name]["tempo_bpm"] = bpm
                new_samples[name]["tempo_conf"] = conf

    if key:
        for track, content in key["results"].items():
            name = _basename(track)
            if name in new_samples:
                new_samples[name]["key"] = parse_key(content)

    return new_samples


def proc_one(reference : pd.Series, root_dir : str):
    print('----')
    print(' >', reference["file_path_midi"])

    midi_obj = miditoolkit.midi.parser.MidiFile(
        Path(root_dir) / reference["file_path_midi"]
    )

    variations = list(itertools.product(transpose_intervals, stretch_factors))

    wav_signals = []
    new_samples = {}

    for (interval, factor) in variations:

        start_time = time.perf_counter()
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

        print(sample_signature_name)

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

        # # convert to nlm
        # nlm_new_path = convert(midi_new_path, os.path.dirname(midi_new_path), root_dir, "midi_to_nlm")
        # nlm_new_path = root_dir / Path(nlm_new_path).relative_to("/data")
        # # convert to wav
        # wav_new_path = convert(midi_new_path, os.path.dirname(midi_new_path), root_dir, "midi_to_wav")
        # wav_new_path = root_dir / Path(wav_new_path).relative_to("/data")

        # wav_signals.append(wav_new_path)

        # print(midi_new_path, nlm_new_path, wav_new_path)

        # duration = get_wav_duration(wav_new_path)

        end_time = time.perf_counter()
        elapsed_time = int(end_time - start_time)
    
        # Mathematical conversion to H:M:S
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Display formatted result
        print(f"Execution time: {hours:02}:{minutes:02}:{seconds:02}")

        # new_sample = reference.copy()
        # new_sample["file_path_midi"] = Path(midi_new_path).relative_to(root_dir)
        # new_sample["file_path_nlm"] = Path(nlm_new_path).relative_to(root_dir)
        # new_sample["file_path_wav"] = Path(wav_new_path).relative_to(root_dir)
        # new_sample["file_name_prefix"] = reference["file_name_prefix"] + sample_signature_name
        # new_sample["duration"] = duration
        # new_sample["has_vgm"] = False
        # new_sample["file_path_midi_tokens"] = None
        # new_sample["file_path_nlm_tokens"] = None

        # new_sample["has_music_track_description"] = None
        # new_sample["file_path_music_track_description"] = None

        # new_samples[wav_new_path] = new_sample

    # create descriptors
    # tempo = detect_tempo(wav_signals, out_dir="/nesmdb/TempoDetector")
    # key = recognize_key(wav_signals, out_dir="/nesmdb/KeyRecognition")

    # # apply the keys
    # # tempo_bpm,tempo_conf,key
    # new_samples = apply_results(new_samples, tempo=tempo)
    # new_samples = apply_results(new_samples, key=key)

    # return list(new_samples.values)



if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='augment.py')
    parser.add_argument('--dataset-path', type=str, required=True)
    args = parser.parse_args()

    root_dir = Path(args.dataset_path).absolute()

    df = pd.read_csv(root_dir / "nes_metadata.csv")
    df["id"] = df.id.map('{:03d}'.format)
    df = df[~(df.split == "test")]
    df = df[df.has_midi]

    print(root_dir)

    # collect
    data = []
    for idx, sample in df.iterrows():
        # append
        data.append([sample, root_dir])
        # if idx > 20:
            # break

    # data = [[
    #     df[(df.id == "005") & (df.track_name == "Ending")].iloc[0],
    #     root_dir
    # ]]

    # data = [[
    #     df[(df.id == "000") & (df.track_name == "GameStart")].iloc[0],
    #     root_dir
    # ]]

    # augmented_data = []
    # for sample in tqdm(data):
    #     new_samples = proc_one(*sample)
    #     augmented_data.extend(new_samples)
    # run, multi-thread
    pool = mp.Pool()
    pool.starmap(proc_one, data)
