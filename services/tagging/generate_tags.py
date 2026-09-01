# this code was adapt from the script at [MidiCaps](https://github.com/AMAAI-Lab/MidiCaps/tree/main) 
# to the context of NES

from collections import Counter
import copy
from pathlib import Path

from music21 import midi
import os
from time import time
from chord_extractor.extractors import Chordino
import json
import argparse
import numpy as np
import mido
import pandas as pd

import logging
from tqdm import tqdm

LOGGING_SPECS = {
    "name":"Tagger", 
    "log_file":"logs/tagging.log", 
    "level":logging.DEBUG
}


def setup_logger(name, log_file=None, level=logging.INFO, queue=None):
    """Sets up a logger that writes messages to a file."""
    
    # 1. Create a custom logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter('[%(asctime)s] %(name)s [%(levelname)s] %(message)s')
    # 2. Create console handlers
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level) 
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    # support for multiprocesing logging
    if queue:
        logger.addHandler(queue)

    return logger


def format_hms(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def get_final_inst_list(midi_file_path):
    INSTRUMENT_NAMES = {
        "p1": "Pulse Wave Generator 1",
        "p2": "Pulse Wave Generator 2",
        "tr": "Triangle Wave Generator",
        "no": "Noise Generator",
    }
    instruments = []
    midi = mido.MidiFile(midi_file_path)
    for track in midi.tracks:
        if track.name in INSTRUMENT_NAMES:
            instruments.append(INSTRUMENT_NAMES[track.name])    
    return instruments


def read_midi(path):
    mf = midi.MidiFile()
    mf.open(path)
    mf.read()
    mf.close()
    return midi.translate.midiFileToStream(mf)

def get_keys(midi_obj):
    # NES specific 
    midi_without_percussion = copy.deepcopy(midi_obj)   # keep the original intact

    for part in list(midi_without_percussion.parts):
        if part.partName == 'no':          # match the track/part name
            midi_without_percussion.remove(part) # remove the percurssion 

    keys = midi_without_percussion.analyze('keys')
    return keys

def get_time_signature(midi):
    timeSignature = midi.getTimeSignatures()[0]
    return timeSignature

def get_tempo(midifile):
    mid = mido.MidiFile(midifile)
    try:
        for msg in mid:
            if msg.type == 'set_tempo':
                tempo = mido.tempo2bpm(msg.tempo)
                return tempo
    except:
        return None
    return None

def get_duration(file):
    try:
        mid = mido.MidiFile(file)
        return mid.length
    except:
        return-1

def find_most_repeating_sequence(chords_list, sequence_length):
    sequences = [tuple(chords_list[i:i+sequence_length]) for i in range(len(chords_list) - sequence_length + 1)]

    delete_index=[]
    for i,seqs in enumerate(sequences):
        if seqs[0] == seqs[-1]:
            delete_index.append(i)
    for i in reversed(delete_index):
        sequences.pop(i)
    sequence_counts = Counter(sequences)
    try:
        most_common_sequence, count = sequence_counts.most_common(1)[0]
        return most_common_sequence, count
    except Exception as e:
        print(e)
        return None, 0

def give_me_final_seq(chords):
    sequence_3, count_3 = find_most_repeating_sequence(chords, 3)
    sequence_4, count_4 = find_most_repeating_sequence(chords, 4)
    sequence_5, count_5 = find_most_repeating_sequence(chords, 5)
    total_count=count_3+count_4+count_5
    if count_5>0.25*(total_count):
        if count_5>0.79*count_4:
            return sequence_5, count_5
    if count_4>0.3*(total_count):
        if count_4>0.79*count_3:
            return sequence_4, count_4
    if count_3==0:
        if count_4==0:
            if count_5==0:
                return None, 0 # everything is 0
            else:
                return sequence_5, count_5
        else:
            return sequence_4, count_4
    else:
        return sequence_3, count_3
    
def parse_tags(tags: dict) -> str:

    def _stringfy(field):
        return ", ".join(map(str, field))

    # ensure None fields are strings
    tags = {k: ("" if v is None else v) for k, v in tags.items()}

    return "; ".join((
        tags["file_game_name"],
        "-".join(tags["genre-wiki"]),
        tags["key"],
        _stringfy(tags["tempo"]),
        tags["time_signature"],
        _stringfy(tags["duration"]),
        str(tags["instruments"]),
        _stringfy(tags["chord_summary"]),
        str(tags["chords"]),
    ))

def generate_tags(track_metadata : pd.Series, dataset_path : Path, chord_estimator):
    file =  Path(dataset_path) / track_metadata["file_path_midi"]
    audio_file = Path(dataset_path) / track_metadata["file_path_wav"]
    music_description_file = Path(dataset_path) / track_metadata["id"] / track_metadata["track_name"] / (track_metadata["file_name_prefix"] + "_music_track_description.md")

    logger = logging.getLogger(LOGGING_SPECS["name"])

    logger.info("Extracting {}".format(os.path.basename(file)))
    midi = read_midi(file)        

    if (not audio_file or not os.path.exists(audio_file)) or \
        (not file or not os.path.exists(file)):
        logger.warning("Files of {} wasnt detected".format(track_metadata["file_name_prefix"]))
        return 
    
    # AUDIO Features
    chords = chord_estimator.extract(audio_file)
    chords_out = [(x.chord, x.timestamp) for x in chords[1:-1]]

    # chord summary
    ch_name=[]
    ch_time=[]

    for ch in chords_out:
        ch_name.append(ch[0])
        ch_time.append(ch[1])
    if len(ch_name)<3:
        final_seq=ch_name
        final_count=1
    else:
        final_seq, final_count = give_me_final_seq(ch_name)
    if final_seq is not None:
        if len(final_seq)==4:
            if final_seq[0]==final_seq[2] and final_seq[1]==final_seq[3]:
                final_seq=final_seq[0:2]
    chord_summary=[final_seq,final_count]

    # MIDI features
    try:
        fulllist=get_final_inst_list(file)
    except Exception as e:
        logger.warning(e)
        logger.warning('skipped instruments')
        fulllist=[]

    #key
    try:
        res_key = get_keys(midi)
        key = res_key.tonic.name + " " + res_key.mode
    except:
        key = None
    #key postprocessing
    if key is None:
        key=key
    elif '-' in key:
        key=key.replace('-','b')
    else:
        key=key
    #time signature
    try:
        time_sig = get_time_signature(midi)
        time_signature = str(time_sig.numerator)+'/'+str(time_sig.denominator)
    except:
        time_signature = None
    #tempo
    try:
        bpm = get_tempo(file)
    except: 
        bpm = None
    #tempo postprocessing
    bpm=np.round(bpm)
    if np.isnan(bpm):
        cap=''
        bpm=''
    else:
        bpm=int(bpm)
        dice=np.random.randint(0,2)
        if dice==0:
            tempo_marks=np.array((40, 60, 70, 90, 110, 140, 160, 210))
            tempo_caps=['Grave', 'Largo', 'Adagio', 'Andante', 'Moderato', 'Allegro', 'Vivace', 'Presto', 'Prestissimo']
            index=np.sum(bpm>tempo_marks)
            tempo_cap=tempo_caps[index]
        else:
            tempo_marks=np.array((80, 120, 160))
            tempo_caps=['Slow', 'Moderate tempo', 'Fast', 'Very fast']
            index=np.int_(np.sum(bpm>tempo_marks))
            tempo_cap=tempo_caps[index]
    #duration
        try:
            duration = get_duration(file)
        except:
            duration = None

    #duration postprocessing
    dur_marks=np.array((30, 120, 300))
    dur_caps=['Short fragment', 'Short song', 'Song', 'Long piece']
    dur=int(np.round(duration))
    index=np.int_(np.sum(dur>dur_marks))
    dur_cap=dur_caps[index]
    #sort nicely
    tags={}
    tags['file_game_name']=track_metadata["file_game_name"]
    tags['genre-wiki'] = eval(track_metadata['genre_wiki']) if track_metadata["genre_wiki"] is not np.nan else []
    tags['console'] = "NES"
    tags['key']=key
    tags['tempo']=[bpm,tempo_cap]
    tags['time_signature'] = time_signature
    tags['duration']=[dur,dur_cap]
    tags['instruments']=fulllist
    tags['chord_summary']=chord_summary
    tags['chords']=chords_out

    tags_description = parse_tags(tags)

    with open(music_description_file, "w") as f:
        f.write(tags_description)

    tags_target_file = music_description_file.with_name(track_metadata["file_name_prefix"] + "_tags.json")
    with open(tags_target_file, "w") as f:
        json.dump(tags, f)

    logger.info("Description created at {}".format(music_description_file))
    logger.info("Tags sucessufully created at {}".format(tags_target_file))


def main():
        # Parse arguments
    parser = argparse.ArgumentParser(description='generate_tags.py')
    parser.add_argument('--dataset-path', type=str, required=True)
    args = parser.parse_args()

    start_time = time()
    dataset_path = Path(args.dataset_path).absolute()

    # setup for multiprocessing logging
    logger = setup_logger(**LOGGING_SPECS)

    chord_estimator = Chordino()

    df = pd.read_csv(dataset_path / "nes_metadata.csv", dtype={"id": str})
    df = df[df.has_midi]
    df = df[df.duration_s <= 60]

    print(dataset_path)
    for (_, sample) in tqdm(df.iterrows(), desc="Tags generation "):
        generate_tags(sample, dataset_path, chord_estimator)

    elapsed = time() - start_time
    logger.info(f"Tag generation completed in {format_hms(elapsed)} seconds")
    print(f"Tag generation completed in {format_hms(elapsed)} seconds")

    # Extracting 044_CastleofDragon_15_16Unused.mid
    # 045_Castlevania_00_01Prologue_#_1_original.mid



if __name__ == "__main__":
    main()  
