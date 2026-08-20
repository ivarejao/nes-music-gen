import logging
from any2music.audio.decoders.musicgen import MusicGenSize

from src.audio.train_audio import train_audio
from src.symbolic import train_symbolic, SYMBOLIC_TYPE
from src.dataset.snes_datasets import NESAudioDataset
from src.dataset.mtg_jamendo_dataset import MTGJamendoAudioDataset

logging.basicConfig(level=logging.INFO)

DATASET_ROOT_NES = "/home/es119256/dados/datasets/nes_dataset"
DATASET_ROOT_JAMENDO = '/home/es119256/dados/datasets/tokenized_mtg_jamendo'

if __name__ == "__main__":
    train_audio(
        dataset_class=MTGJamendoAudioDataset,
        dataset_root = DATASET_ROOT_JAMENDO,
        epochs = 100,
        warmup_steps = 350,
        audio_secs = 60,
        decoder_size = MusicGenSize.SMALL,
        checkpoint_every = 5
    )

    # train_audio(
    #     dataset_class=NESAudioDataset,
    #     dataset_root = DATASET_ROOT_NES,
    #     epochs = 100,
    #     warmup_steps = 350,
    #     audio_secs = 60,
    #     decoder_size = MusicGenSize.SMALL,
    #     checkpoint_every = 5
    # )

    # train_symbolic(
    #     type = SYMBOLIC_TYPE.MIDI,
    #     dataset_root = DATASET_ROOT,
    #     epochs = 100,
    #     warmup_steps = 350,
    #     decoder_size = MusicGenSize.SMALL,
    #     checkpoint_every = 5
    # )

    # train_symbolic(
    #     type = SYMBOLIC_TYPE.NLM,
    #     dataset_root = DATASET_ROOT,
    #     epochs = 100,
    #     warmup_steps = 350,
    #     decoder_size = MusicGenSize.SMALL,
    #     checkpoint_every = 5
    # )