import os
import logging
import random
import csv

import pandas as pd

import torch

from audiotools import AudioSignal

from any2music.audio.decoders.musicgen import DelayProvider
from any2music.text.encoders import T5Conditioner
from any2music.audio.tokenizers import DACCompressionModel
from any2music.audio.decoders.musicgen import MusicGenSize, MUSICGEN_SIZES
from any2music.audio.dataloaders.mtg_jamendo import MTGJamendoDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_ROOT = "/home/es119256/dados/datasets/nes_dataset"

def select_samples(n, split, path):
    random.seed(42)

    dataset = MTGJamendoDataset(
        dataset_root='/home/es119256/dados/datasets/mtg-jamendo',
        mtg_clone='/home/es119256/dados/repos/passt-on-mtg-jamendo-dataset',
        split_type=split
    )

    if n > len(dataset):
        raise ValueError("n must be less or equal to dataset size")

    split_csv = [['path', 'duration', 'desc']]
    idx_choices = random.choices(range(len(dataset)), k=n)

    for idx in idx_choices:
        example = dataset[idx]
        split_csv.append(example.values())

    with open(path, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(split_csv)

    return path


def compress_reconstruct(file_path_wav:str, audio_secs:int, audio_tokenizer):
    max_audio_len = audio_tokenizer.sample_rate*audio_secs

    # Keep preprocessing on CPU so dataloader workers can safely serialize results.
    # DAC compress function already resamples
    audio_tensor = AudioSignal(file_path_wav, device='cuda').to_mono()[:, :, :max_audio_len].cuda()

    with torch.no_grad():
        # Tokenize the audio
        encoded_audio, meta = audio_tokenizer.encode(audio_tensor)
        decoded_audio = audio_tokenizer.decode(encoded_audio, meta).cpu()

    return audio_tensor.cpu(), decoded_audio


def run(
        csv_path:str,
        save_folder:str,
        audio_secs:int=60
    ):
    original_folder = os.path.join(save_folder, 'original')
    reconstruct_folder = os.path.join(save_folder, 'reconstruct')

    for folder in [original_folder, reconstruct_folder]:
        if not os.path.isdir(folder):
            os.makedirs(folder, exist_ok=True)

    # Get df
    df = pd.read_csv(csv_path)

    # Get DAC
    dac = DACCompressionModel.get_pretrained("44khz", device='cuda').eval().cuda()
    dac.set_num_codebooks(4)

    for row in df.to_dict(orient='records'):
        wav_path = row["path"]

        file_name_prefix  = wav_path.split('/')[-1][:-4]
        save_path = os.path.join(save_folder, file_name_prefix + '_pre_comp.pt')

        # Compress and recosntruct audio
        original, reconstructed = compress_reconstruct(wav_path, audio_secs, dac)

        original.write(os.path.join(original_folder, file_name_prefix + '.wav'))
        reconstructed.write(os.path.join(reconstruct_folder, file_name_prefix + '.wav'))


if __name__ == '__main__':
    # Get splits with the same size as in the NES dataset
    train_csv = select_samples(n=4196, split='train', path='./mtg_jamendo_train.csv')
    test_csv = select_samples(n=334, split='test', path='./mtg_jamendo_test.csv')

    # Save copy from the original. Compress and reconstruct with DAC and save reconstructed audios
    run(train_csv, save_folder='/home/es119256/dados/datasets/reconstruct_mtg_jamendo/train')
    run(test_csv, save_folder='/home/es119256/dados/datasets/reconstruct_mtg_jamendo/test')

