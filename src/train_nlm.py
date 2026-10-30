import logging
from any2music.audio.decoders.musicgen import MusicGenSize


from src.symbolic import train_symbolic, SYMBOLIC_TYPE

logging.basicConfig(level=logging.INFO)

DATASET_ROOT_NES = "/home/es119256/dados/datasets/nes_dataset"

if __name__ == "__main__":
    train_symbolic(
        type=SYMBOLIC_TYPE.NLM,
        dataset_root=DATASET_ROOT_NES,
        epochs=70,
        warmup_steps=3_000,
        generate_every=10,
        early_stop_patience=5,
        decoder_size=MusicGenSize.SMALL
    )