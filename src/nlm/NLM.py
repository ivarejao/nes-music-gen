# NLM.py

import os
import pickle
import requests
from pathlib import Path
from argparse import Namespace, ArgumentParser
from urllib.parse import urljoin

import numpy as np
from traitlets import Unicode

from src.nlm.tokenizer import NLMTokenizer
from src.utils import setup_logger

#API_URL = "http://localhost:5555"
API_URL = "http://127.0.0.1:5555"


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--tokenizer-path', help='Define tokenizer path')
    parser.add_argument('--input-nlm',      help='Define input .nlm.pkl file path')
    parser.add_argument('--output-dir',     help='Define output directory')
    return parser.parse_args()

# Valid nlm command tokens -> {'reg': register offset, 'max': max value}.
# Function tokens are '<ch>_<fu>'. Special commands: 'w' (wait), 'clock'.
# 'reg'/'max' are None for special commands (not APU register-backed).
NLM_COMMANDS = {
    # --- Pulse 1 ---
    'p1_du': {'reg': 0, 'max': 3},
    'p1_lh': {'reg': 0, 'max': 1},
    'p1_cv': {'reg': 0, 'max': 1},
    'p1_vo': {'reg': 0, 'max': 15},
    'p1_se': {'reg': 1, 'max': 1},
    'p1_sp': {'reg': 1, 'max': 7},
    'p1_sn': {'reg': 1, 'max': 1},
    'p1_ss': {'reg': 1, 'max': 7},
    'p1_tl': {'reg': 2, 'max': 255},
    'p1_ll': {'reg': 3, 'max': 31},
    'p1_th': {'reg': 3, 'max': 7},

    # --- Pulse 2 ---
    'p2_du': {'reg': 0, 'max': 3},
    'p2_lh': {'reg': 0, 'max': 1},
    'p2_cv': {'reg': 0, 'max': 1},
    'p2_vo': {'reg': 0, 'max': 15},
    'p2_se': {'reg': 1, 'max': 1},
    'p2_sp': {'reg': 1, 'max': 7},
    'p2_sn': {'reg': 1, 'max': 1},
    'p2_ss': {'reg': 1, 'max': 7},
    'p2_tl': {'reg': 2, 'max': 255},
    'p2_ll': {'reg': 3, 'max': 31},
    'p2_th': {'reg': 3, 'max': 7},

    # --- Triangle ---
    'tr_lh': {'reg': 0, 'max': 1},
    'tr_lr': {'reg': 0, 'max': 127},
    'tr_tl': {'reg': 2, 'max': 255},
    'tr_ll': {'reg': 3, 'max': 31},
    'tr_th': {'reg': 3, 'max': 7},

    # --- Noise ---
    'no_lh': {'reg': 0, 'max': 1},
    'no_cv': {'reg': 0, 'max': 1},
    'no_vo': {'reg': 0, 'max': 15},
    'no_nl': {'reg': 2, 'max': 1},
    'no_np': {'reg': 2, 'max': 15},
    'no_ll': {'reg': 3, 'max': 31},

    # --- Channel status ---
    'ch_no': {'reg': 0, 'max': 1},
    'ch_tr': {'reg': 0, 'max': 1},
    'ch_p2': {'reg': 0, 'max': 1},
    'ch_p1': {'reg': 0, 'max': 1},

    # --- Frame counter ---
    'fc_mo': {'reg': 0, 'max': 1},

    # --- Special (non-register) commands ---
    'w':     {'reg': None, 'max': None},   # wait / time advance
    'clock': {'reg': None, 'max': None},   # clock rate command
}

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

    def parse_nlm(self, nlm_text: str) -> list[tuple]:
        """Validate a loaded nlm structure against NLM_COMMANDS.

        An nlm is: [header, comm, comm, ...] where each comm is a 2-tuple:
        ('w', duration)          -> wait
        ('clock', rate)          -> clock header/command
        ('<ch>_<fu>', value)     -> APU function, value must be 0..max

        Returns a list of human-readable error strings (empty list == valid).
        """
        cmds_text = nlm_text.split()

        valids_commands = []

        for i, comm in enumerate(cmds_text):
            parts = comm.rsplit("_", 1)
            if len(parts) != 2:
                self.logger.warning(
                    'index {}: expected a 2-element command, got {!r}'.format(i, comm))
                continue

            token, value = parts[0], parts[1]

            try:
                # register-backed function: value must be int within 0..max.
                value = int(value)
            except ValueError:
                self.logger.warning(
                    'index {}: command {!r} expects an integer value, got {!r}'.format(
                        i, token, value))
                continue

            # Normalize unicode (py2.7) so 'w' == u'w' during lookup.
            key = token.encode('utf-8').decode('utf-8') if isinstance(token, Unicode) else token

            if key not in NLM_COMMANDS:
                self.logger.warning(
                    'index {}: unknown command token {!r}'.format(i, token))
                continue

            spec = NLM_COMMANDS[key]

            # Special commands (w, clock): only require an int-ish value.
            if spec['max'] is None:
                if not isinstance(value, int):
                    self.logger.warning(
                        'index {}: command {!r} expects an integer value, got {!r}'.format(
                            i, token, value))
                    continue
            else:
                # Regular commands: check if value is within limits MODIFICAÇÃO DO JOÃO
                if value < 0 or value > spec['max']:
                    self.logger.warning(
                        'index {}: command {!r} value {} out of range 0..{}'.format(
                            i, token, value, spec['max']))
                    continue
            
            valids_commands.append((token, value))


        return valids_commands


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