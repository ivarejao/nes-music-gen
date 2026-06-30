# NLM.py

import os
import pickle
import requests
from pathlib import Path
from argparse import Namespace, ArgumentParser
from urllib.parse import urljoin

import numpy as np

from src.nlm.tokenizer import NLMTokenizer
from src.utils import setup_logger

API_URL = "http://localhost:5555"


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--tokenizer-path', help='Define tokenizer path')
    parser.add_argument('--input-nlm',      help='Define input .nlm.pkl file path')
    parser.add_argument('--output-dir',     help='Define output directory')
    return parser.parse_args()


class NLMHandler():

    def __init__(self, tokenizer_path: str):
        self._tokenizer = NLMTokenizer(tokenizer_path, log_file="log_nlm_tokenizer.log")
        self.logger = setup_logger("NLMHandler", log_file="log_nlm_handler.log")
        self.api_running = self._check_nesmdb_api()
        self.data_mount_point = self.get_nesmdb_mount_point()

    def synthesize(self, input_nlm_path: str | Path, output_dir: str | Path) -> str | None:
        """Convert a .nlm.pkl file to audio via the NESMdb API (nlm_to_wav)."""
        input_nlm_path = Path(input_nlm_path).absolute()
        output_dir     = Path(output_dir).absolute()

        if not self.api_running:
            self.logger.warning("API is offline, skipping synthesis of {}".format(input_nlm_path))
            return None

        if str(self.data_mount_point) not in str(input_nlm_path):
            self.logger.error("Input file is not accessible to docker nesmdb. Should be at {}".format(self.data_mount_point))
        if str(self.data_mount_point) not in str(output_dir):
            self.logger.error("Output dir is not accessible to docker nesmdb. Should be at {}".format(self.data_mount_point))

        # Remap host paths to docker-internal /data paths
        input_nlm_path = Path("/data") / input_nlm_path.relative_to(Path(self.data_mount_point))
        output_dir     = Path("/data") / output_dir.relative_to(Path(self.data_mount_point))

        endpoint = urljoin(API_URL, "convert")
        params = {
            "convertionkey":  "nlm_to_wav",
            "path_to_input":  str(input_nlm_path),
            "output_dir":     str(output_dir),
        }
        response = requests.post(endpoint, json=params)
        if response.status_code == 200:
            output_file_path = response.json()["file_path"]
            self.logger.info("File {} synthesized onto file {}".format(
                os.path.basename(str(input_nlm_path)), output_file_path))
            return output_file_path
        else:
            self.logger.error("Error on synthesis of file {}".format(input_nlm_path))
            return None

    def tokenize(self, nlm_file_path: str | Path) -> np.ndarray:
        """Tokenize a .nlm.pkl file and return the token ids."""
        return self._tokenizer.tokenize(nlm_file_path)

    def detokenize(self, numeric_ids: list | np.ndarray) -> str:
        """Decode numeric token ids back into nlm script text """
        return self._tokenizer.detokenize(numeric_ids)

    def parse_nlm(self, nlm_text : str) -> list[tuple]:
        """Parse the nlm text into original format of pickle file, a list of (function_name, value) tuples."""
        original_tuples = []
        for cmd in nlm_text.split():
            parts = cmd.rsplit("_", 1)
            if len(parts) == 2:
                try:
                    original_tuples.append((parts[0], int(parts[1])))
                except ValueError:
                    pass  # skip malformed tokens

        return original_tuples

    def save_nlm(self, tuples: list[tuple], output_path: str | Path) -> None:
        """Persist a list of (function_name, value) tuples as a .nlm.pkl file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(tuples, f, protocol=2) # protocol 2 for nesmdb compability
        self.logger.info("NLM data saved at {}".format(output_path))

    def _check_nesmdb_api(self, timeout: int = 5) -> bool:
        try:
            requests.get(API_URL, timeout=timeout)
            self.logger.info(f"API is running at {API_URL}")
            return True
        except (requests.ConnectionError, requests.Timeout):
            self.logger.warning(f"Synthesis is not available: API offline at {API_URL}")
            return False

    def get_nesmdb_mount_point(self) -> str | None:
        if not self.api_running:
            self.logger.warning("API is offline, skipping getting mount point")
            return None
        endpoint = urljoin(API_URL, "data-mount-point")
        response = requests.get(endpoint)
        return response.json()["DATA_MOUNT_POINT"]


if __name__ == "__main__":
    args = parse_args()
    handler = NLMHandler(args.tokenizer_path)

    tokens = handler.tokenize(args.input_nlm)

    nlm_script = handler.detokenize(tokens.ids)
    nlm_tuples = handler.parse_nlm(nlm_script)

    # save reconstructed .nlm.pkl to the mount point as a temp file
    # in order to synthesize
    temp_nlm = Path(handler.data_mount_point) / "temp_sample.nlm.pkl"
    handler.save_nlm(nlm_tuples, temp_nlm)

    assert temp_nlm.exists()

    output_dir = handler.data_mount_point if not args.output_dir else args.output_dir
    converted_file = handler.synthesize(temp_nlm, output_dir)
    converted_file = converted_file.replace("/data", handler.data_mount_point)

    assert os.path.exists(converted_file)

    os.remove(temp_nlm)
    os.remove(converted_file)