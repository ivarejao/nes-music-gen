import logging
from any2music.audio.decoders.musicgen import MusicGenSize

from src.audio import train_audio

logging.basicConfig(level=logging.INFO)

DATASET_ROOT = "/home/es119256/dados/datasets/nes_dataset"

if __name__ == "__main__":
    train_audio(
        dataset_root = DATASET_ROOT,
        epochs = 100,
        warmup_steps = 350,
        audio_secs = 60,
        decoder_size = MusicGenSize.SMALL,
        checkpoint_every = 5
    )