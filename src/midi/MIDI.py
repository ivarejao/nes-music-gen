import subprocess
import os
from src.midi.tokenizer import MIDITokenizer
from symusic.factory import Score
import numpy as np
import requests
from src.utils import setup_logger
from urllib.parse import urljoin
import tempfile
from argparse import Namespace, ArgumentParser
from pathlib import Path

API_URL = "http://localhost:5555"

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--tokenizer-path', help='Define tokenizer path')
    parser.add_argument('--input-midi', help='Define input-midi')
    parser.add_argument('--output-dir', help='Define output-dir')

    args = parser.parse_args()
    return args

class MIDIHandler():

    def __init__(self, tokenizer_path : str):
        self._tokenizer = MIDITokenizer(tokenizer_path, log_file="log_midi_tokenizer.log")
        self.logger = setup_logger("MIDIHandler", log_file="log_midi_handler.log")
        self.api_running = self._check_nesmdb_api()
        self.data_mount_point = self.get_nesmdb_mount_point()

    def synthesize(self, input_midi_path : str | Path, output_dir : str | Path) -> str | None:
        input_midi_path = Path(input_midi_path)
        output_dir = Path(output_dir)

        input_midi_path = input_midi_path.absolute()
        output_dir = output_dir.absolute()

        if not self.api_running:
            self.logger.warning("API is offline, skiping sythesis of {}".format(input_midi_path))
            return None

        if str(self.data_mount_point) not in str(input_midi_path):
            self.logger.error("Input file is not accessible to docker nesmdb. Should be at {}".format(self.data_mount_point))
        if str(self.data_mount_point) not in str(output_dir):
            self.logger.error("Output dir is not accessible to docker nesmdb. Should be at {}".format(self.data_mount_point))

        # convert files to the respectively docker path
        input_midi_path = Path("/data") / input_midi_path.relative_to(Path(self.data_mount_point))
        output_dir = Path("/data") / output_dir.relative_to(Path(self.data_mount_point))
        
        endpoint = urljoin(API_URL, "convert")
        params = {
            "convertionkey" : "midi_to_wav",
            "path_to_input" : str(input_midi_path),
            "output_dir" : str(output_dir)
        }
        response = requests.post(endpoint, json=params)
        if response.status_code == 200:
            output_file_path = response.json()["file_path"]
            self.logger.info("File {} sythesized onto file {}".format(os.path.basename(input_midi_path), output_file_path))
            return output_file_path
        else:
            self.logger.error("Error on synthesis of file {}".format(input_midi_path))
            return None

    def tokenize(self, midi_file_path : str) -> np.array:
        return self._tokenizer.tokenize(midi_file_path)

    def detokenize(self, numeric_ids) -> Score:
        return self._tokenizer.detokenize(numeric_ids)

    def _check_nesmdb_api(self, timeout : int = 5):
        endpoint = urljoin(API_URL, "data-mount-point")
        try:
            requests.get(API_URL, timeout=timeout)
            self.logger.info(f"API is running at {API_URL}")
            return True
        except (requests.ConnectionError, requests.Timeout):
            self.logger.warning(f"Syntesis is not available: API offline at {API_URL}")
            return False

    def get_nesmdb_mount_point(self):
        if not self.api_running:
            self.logger.warning("API is offline, skiping getting mount point")
            return None
        endpoint = urljoin(API_URL, "data-mount-point")
        response = requests.get(endpoint)
        return response.json()["DATA_MOUNT_POINT"]


if __name__ == "__main__":
    args = parse_args()
    mmm = MIDIHandler(args.tokenizer_path)

    tokens = mmm.tokenize(args.input_midi)
    midi_detokenized = mmm.detokenize(tokens.ids)

    input_file = Path(mmm.data_mount_point) / "temp_sample.mid"
    midi_detokenized.dump_midi(input_file)

    assert os.path.exists(input_file)
    
    output_dir = mmm.data_mount_point if not args.output_dir else args.output_dir
    converted_file = mmm.synthesize(input_file, output_dir)
    converted_file = converted_file.replace("/data", mmm.data_mount_point)

    assert os.path.exists(converted_file)

    os.remove(input_file)
    os.remove(converted_file)


