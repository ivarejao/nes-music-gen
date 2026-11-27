import logging
from any2music.audio.decoders.musicgen import MusicGenSize

from src.audio.train_audio import train_audio
from src.dataset.snes_datasets import NESAudioDataset

logging.basicConfig(level=logging.INFO)

DATASET_ROOT_NES = "/home/es119256/dados/datasets/nes_dataset"
DATASET_ROOT_JAMENDO = '/home/es119256/dados/datasets/tokenized_mtg_jamendo'

if __name__ == "__main__":
    train_audio(
        dataset_class=NESAudioDataset,
        dataset_root=DATASET_ROOT_NES,
        epochs=70,
        warmup_steps=3000,
        audio_secs=60,
        generate_every=10,
        early_stop_patience=5,
        decoder_size = MusicGenSize.SMALL
    )