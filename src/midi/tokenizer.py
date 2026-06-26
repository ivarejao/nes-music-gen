from miditok import MMM, TokenizerConfig
from pathlib import Path
import numpy as np
import logging
from argparse import Namespace, ArgumentParser

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--tokenizer-path')

    args = parser.parse_args()
    return args

def setup_logger(name, log_file=None, level=logging.INFO):
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

    return logger

class MIDITokenizer():

    def __init__(self, tokenizer_path : str | None = None, log_file : str | None = None):
        self.tokenizer_path = tokenizer_path
        self.logger = setup_logger("MIDITokenizer", log_file=log_file)
        self._load()

    def _load(self):

        if self.tokenizer_path:
            self.logger.info(f"Loading tokenizer from {self.tokenizer_path}")
            self.model = MMM(params=self.tokenizer_path)
        else:
            self.logger.info("No path provided, creating new tokenizer with default config")
            config = TokenizerConfig(
                pitch_range=(33, 107), 
                beat_res={(0, 1): 8, (1, 2): 4, (2, 4): 2, (4, 8): 1}, 
                num_velocities=16, 
                use_chords=False,    
                use_programs=True,
                use_tempos=True
            )
            config.additional_params["base_tokenizer"] = "REMI"
            self.model = MMM(config)

    def save_model(self, output_dir : str = "midi_tokenizer"):
        # ensure is Path
        output_dir = Path(output_dir)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_params(output_dir)
        self.logger.info("Tokenizer saved at {}".format(output_dir))

    def tokenize(self, midi_file_path : str | Path, **kwargs):
        if not (isinstance(midi_file_path, str) or isinstance(midi_file_path, Path)):
            self.logger.error("For encoding, the model should receive the MIDI file path")
            raise ValueError
        tokens = self.model(midi_file_path, **kwargs)
        return tokens

    def detokenize(self, numeric_ids : list | np.ndarray, **kwargs):
        if not (isinstance(numeric_ids, list) or isinstance(numeric_ids, np.ndarray)):
            self.logger.error("For detokenizing, the model should receive the numeric ids *tokens*")
            raise ValueError
        self.logger.debug(f"Total Tokens read: {len(numeric_ids)}")
        midi_detokenized = self.model(numeric_ids, **kwargs)
        return midi_detokenized



if __name__ == "__main__":
    args = parse_args()
    tokenizer_path = Path(args.tokenizer_path) if args.tokenizer_path else args.tokenizer_path
    midi_tokenizer = MIDITokenizer(tokenizer_path)
    print("Tokenizador MMM (Base REMI) criado com sucesso!")
    print(f"Tamanho do vocabulário fixo: {len(midi_tokenizer.model)} tokens.")

    caminho_salvamento = Path("tokenizers/symbolic/nes_tokenizers/nes_mmm_tokenizer.json")
    midi_tokenizer.save_model(caminho_salvamento)
    print(f"Tokenizador salvo em: {caminho_salvamento}")