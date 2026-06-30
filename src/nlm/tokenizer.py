from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
import pickle
from argparse import Namespace, ArgumentParser
from src.utils import setup_logger
import numpy as np
from pathlib import Path

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--tokenizer-path')

    args = parser.parse_args()
    return args

class NLMTokenizer():
    
    def __init__(self, tokenizer_path : str | None = None, log_file : str | None = None):
        self.tokenizer_path = tokenizer_path
        self.logger = setup_logger("NLMTokenizer", log_file=log_file)
        self._load()

    def _load(self):
        if self.tokenizer_path:
            self.logger.info(f"Loading tokenizer from {self.tokenizer_path}")
            self.model = Tokenizer.from_file(str(self.tokenizer_path))
            self.model.decoder = ByteLevelDecoder()
        else:
            self.logger.error("No path to tokenizer config was provided")

    def save_model(self, output_dir : str = "nlm_tokenizer"):
        # ensure is Path
        output_dir = Path(output_dir)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(output_dir))
        self.logger.info("Tokenizer saved at {}".format(output_dir))

    def tokenize(self, nlm_file_path : str | Path, **kwargs):
        '''
            nlm_file_path : (.nlm.pkl) file
        '''
        if not (isinstance(nlm_file_path, str) or isinstance(nlm_file_path, Path)):
            self.logger.error("For encoding, the model should receive the MIDI file path")
            raise ValueError
        with open(nlm_file_path, "rb") as f:
            nlm_data = pickle.load(f)
            nlm_script = " ".join([f"{cmd[0]}_{cmd[1]}" for cmd in nlm_data])
        tokens = self.model.encode(nlm_script, **kwargs)
        return tokens

    def detokenize(self, numeric_ids : list | np.ndarray, **kwargs) -> str:
        if not (isinstance(numeric_ids, list) or isinstance(numeric_ids, np.ndarray)):
            self.logger.error("For detokenizing, the model should receive the numeric ids *tokens*")
            raise ValueError
        self.logger.debug(f"Total Tokens read: {len(numeric_ids)}")
        nlm_text = self.model.decode(numeric_ids, **kwargs)

        return nlm_text

if __name__ == "__main__":
    args = parse_args()
    tokenizer_path = Path(args.tokenizer_path) if args.tokenizer_path else args.tokenizer_path
    nlm_tokenizer = NLMTokenizer(tokenizer_path)
    import ipdb
    ipdb.set_trace()
    print("Tokenizador BPE (Base ByteLevel) criado com sucesso!")
    print(f"Tamanho do vocabulário fixo: {len(nlm_tokenizer.model.get_vocab())} tokens.")

    caminho_salvamento = Path("models/tokenizers/symbolic/test_nlm_bpe_373_tokenizer.json")
    nlm_tokenizer.save_model(caminho_salvamento)
    print(f"Tokenizador salvo em: {caminho_salvamento}")