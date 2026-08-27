#
# Clean MIDI data to:
# 1) Remove all pieces that are not in common time (i.e., 4/4)
# 2) Remove all non-piano tracks
# 3) Combine multiple piano tracks into a single one
# TODO: Organize

import logging
import logging.handlers
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

from src.utils import setup_logger




DESCRIPTOR_API_URL = "http://localhost:8000"
NESMDB_API_URL = "http://localhost:5555"

LOGGING_SPECS = {
    "name":"Augment", 
    "log_file":"augment.log", 
    "level":logging.DEBUG
}


# Range of transposition intervals (in semi-tones)
# transpose_intervals = range(-4, 5)

# # Range of strech factors
# stretch_factors = [0.8, 0.9, 1.0]

# debug
transpose_intervals = range(-1, 2)  
stretch_factors = [0.8, 1.0, 1.1]

class TrackAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return "[%s] %s" % (self.extra["track"], msg), kwargs

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
    # print("Wav signal cropped on {}".format(output_path))


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

    logger = logging.getLogger(LOGGING_SPECS["name"])

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
        logger.info("File {} sythesized onto file {}".format(os.path.basename(input_midi_path), output_file_path))
        return output_file_path
    else:
        logger.info("Error on synthesis of file {}\n{}".format(input_midi_path, response.json()["error"]))
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
    
    # print(tracks)
    # print(payload, f"{DESCRIPTOR_API_URL}/{endpoint}")
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


def worker(reference : pd.Series, root_dir : str, log_queue : mp.Queue):

    # logger = setup_logger(**LOGGING_SPECS, queue=log_queue)

    queue_handler = logging.handlers.QueueHandler(log_queue)
    base_logger = logging.getLogger(LOGGING_SPECS["name"])
    base_logger.handlers.clear()        # drop inherited console+file handlers
    base_logger.setLevel(logging.INFO)
    base_logger.addHandler(queue_handler)

    logger = TrackAdapter(base_logger, {"track": reference["id"] + "_" + reference["track_name"]})

    logger.info("Starting augmentation of " + str(reference["file_path_midi"]))

    midi_obj = miditoolkit.midi.parser.MidiFile(
        Path(root_dir) / reference["file_path_midi"]
    )

    variations = list(itertools.product(transpose_intervals, stretch_factors))

    wav_signals = []
    new_samples = {}

    for (interval, factor) in variations:
        if interval == 0 and factor == 1.0:
            # We dont need this variation
            continue

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

        track_signature_name = interval_suffix + strech_suffix

        logger.info("Track signature: " + track_signature_name)

        # create directory
        base_dir = Path(
            os.path.join(root_dir, str(reference["id"]), reference["track_name"] + track_signature_name)
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

        midi_new_path = base_dir / (reference["file_name_prefix"] + track_signature_name + ".mid")

        # save midi
        logger.debug("New MIDI Track: " + str(midi_new_path))
        transposed_streched_mid.dump(midi_new_path)

        # convert to nlm
        nlm_new_path = convert(midi_new_path, os.path.dirname(midi_new_path), root_dir, "midi_to_nlm")
        nlm_new_path = root_dir / Path(nlm_new_path).relative_to("/data")
        # convert to wav
        wav_new_path = convert(midi_new_path, os.path.dirname(midi_new_path), root_dir, "midi_to_wav")
        wav_new_path = root_dir / Path(wav_new_path).relative_to("/data")

        wav_signals.append(wav_new_path)

        logger.debug("New paths: \n" + str(midi_new_path) + " | " + str(nlm_new_path) + " | " + str(wav_new_path))

        duration = get_wav_duration(wav_new_path)

        end_time = time.perf_counter()
        elapsed_time = int(end_time - start_time)
    
        # Mathematical conversion to H:M:S
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Display formatted result
        logger.info(f"Execution time: {hours:02}:{minutes:02}:{seconds:02}")

        augmented_sample = reference.copy()
        augmented_sample["file_path_midi"] = Path(midi_new_path).relative_to(root_dir)
        augmented_sample["file_path_nlm"] = Path(nlm_new_path).relative_to(root_dir)
        augmented_sample["file_path_wav"] = Path(wav_new_path).relative_to(root_dir)
        augmented_sample["file_name_prefix"] = reference["file_name_prefix"] + track_signature_name
        augmented_sample["duration"] = duration
        augmented_sample["has_vgm"] = False
        augmented_sample["file_path_midi_tokens"] = None
        augmented_sample["file_path_nlm_tokens"] = None

        augmented_sample["has_music_track_description"] = None
        augmented_sample["file_path_music_track_description"] = None

        new_samples[wav_new_path] = augmented_sample

    # convert the wav signals path to the respective file system 
    # of the docker service
    wav_container_paths = [Path("/data/") / p.relative_to(root_dir) for p in wav_signals]

    # create descriptors in batch
    tempo = detect_tempo(wav_container_paths, out_dir="/data/TempoDetector")
    key = recognize_key(wav_container_paths, out_dir="/data/KeyRecognition")

    # apply the keys
    # tempo_bpm,tempo_conf,key
    new_samples = apply_results(new_samples, tempo=tempo)
    new_samples = apply_results(new_samples, key=key)

    # then save it the metadata for each new track, in order to make 
    # medium results
    for wav_path, metadata in new_samples.items():
        metadata = metadata.to_frame().transpose()
        wav_path_name = wav_path.name.split('.')[0]
        track_metadata_path = wav_path.with_name(f"{wav_path_name}.csv")
        track_signature_name = os.path.basename(os.path.dirname(track_metadata_path)) 
        metadata.to_csv(track_metadata_path, index=False)
        logger.info("Saving metadata of track {} at\n |-> {}".format(track_signature_name, track_metadata_path))
    
    logger.info("Sucessful augmented of {} into {} new samples".format(os.path.basename(reference["file_path_midi"]), len(new_samples)))



if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='augment.py')
    parser.add_argument('--dataset-path', type=str, required=True)
    args = parser.parse_args()

    root_dir = Path(args.dataset_path).absolute()

    # setup for multiprocessing logging
    logger = setup_logger(**LOGGING_SPECS)
    manager = mp.Manager()
    log_queue = manager.Queue() # specific queue that will be responsible for logging into file

    listener = logging.handlers.QueueListener(
        log_queue, 
        *logger.handlers,
        respect_handler_level=True
    )
    listener.start()

    # TODO: add NLMHandler and MIDIHandler loggers into the main logger

    df = pd.read_csv(root_dir / "nes_metadata.csv")
    df["id"] = df.id.map('{:03d}'.format)
    df = df[~(df.split == "test")] # TODO: remove this when running
    df = df[df.has_midi]
    df = df[df.duration_s <= 60]

    print(root_dir)
    logger.info("Augmentation of {} tracks".format(df.shape[0]))

    # collect
    data = []
    for idx, sample in df.iterrows():
        # append
        data.append([sample, root_dir, log_queue])
        # if idx > 20:
            # break

    # data = [[
    #     df[(df.id == "005") & (df.track_name == "Ending")].iloc[0],
    #     root_dir
    # ]]

    # data = [
    #     (
    #         df[(df.id == "000") & (df.track_name == "GameStart")].iloc[0],
    #         root_dir
    #     ),
    #     (
    #         df[(df.id == "005") & (df.track_name == "Ending")].iloc[0],
    #         root_dir
    #     )
    # ]

    

    # proc_one(*data[0])
    # augmented_data = []
    # for sample in tqdm(data):
    #     new_samples = proc_one(*sample)
    #     augmented_data.extend(new_samples)
    # run, multi-thread
    pool = mp.Pool(processes=10) # NOTE: Hardcoded for g15 notebook
    pool.starmap(worker, data)
