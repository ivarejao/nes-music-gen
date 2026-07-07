import os
import random
import logging
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset
from torchaudio import save as save_audio

from audiotools import AudioSignal
from any2music.audio.tokenizers import DACCompressionModel, HFEncodecCompressionModel
from any2music.audio.utils import load_mono_and_resample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_ROOT = "/home/es119256/dados/repos/nes-music-gen/my_xp_jamendo"

class JamendoDataset(Dataset):
    def __init__(self, dataset_root:str) -> None:
        super().__init__()
        self.dataset_root = dataset_root
        #self.sample_songs()
        self.list_songs()

        logger.info(self.files_list)

    def list_songs(self):
        files = []

        for file in os.listdir(self.dataset_root):
            file_path = os.path.join(self.dataset_root, file)
            files.append(file_path)

        logger.info(files)

        self.files_list = files

    def sample_songs(self):
        files = []

        for folder in os.listdir(self.dataset_root):
            folder_path = os.path.join(DATASET_ROOT, folder)

            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                files.append(file_path)

        chosen_files = random.choices(files, k=334)
        self.files_list = chosen_files

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, index):
        return self.files_list[index]

def instantiate_xp(xp_dir):
    xp_folder = os.path.join(xp_dir, "test_audio_tokenizers_jamendo")
    encodec_folder = os.path.join(xp_folder, "encodec")
    dac_folder = os.path.join(xp_folder, "dac")

    if not os.path.isdir(xp_folder):
        os.makedirs(xp_folder, exist_ok=True)

        for subfolder in [encodec_folder, dac_folder]:
            os.makedirs(subfolder, exist_ok=True)

    return xp_folder, encodec_folder, dac_folder


def tokenize(wav:str, audio_secs:int, audio_tokenizer):
    sr = audio_tokenizer.sample_rate
    max_audio_len = sr*audio_secs
    max_audio_tokens = int(audio_tokenizer.frame_rate*audio_secs)

    audio_tensor = AudioSignal(wav, device='cuda').resample(sr).to_mono()[:, :, :max_audio_len].cuda()

    # Encodec needs the Tensor not the AudioSignal
    if sr == 32_000:
        audio_tensor = audio_tensor.audio_data

    # Tokenize the audio
    with torch.no_grad():
        encoded_audio, meta = audio_tokenizer.encode(audio_tensor)
        encoded_audio = encoded_audio[:, :, :max_audio_tokens]

    if hasattr(audio_tensor, 'audio_data'):
        audio_tensor = audio_tensor.audio_data

    return encoded_audio, meta, audio_tensor


def detokenize(encoded_audio, meta, audio_tokenizer):
    decoded_audio = audio_tokenizer.decode(encoded_audio, meta).cpu()
    return decoded_audio


def test_audio_tokenizers(
        dataset_root:str, 
        audio_secs:int=60,
        xp_dir = './',
        convert=False
    ):

    # Instanciate xp
    xp_folder, encodec_folder, dac_folder = instantiate_xp(xp_dir)

    # Get models
    dac = DACCompressionModel.get_pretrained("44khz", device='cpu').eval().cuda()
    dac.set_num_codebooks(4)

    encodec = HFEncodecCompressionModel.get_pretrained('facebook/encodec_32khz').cuda()

    # Get dataloader
    BATCH_SIZE = 12
    test_dataset = JamendoDataset(dataset_root)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    dac_error = 0
    encodec_error = 0
    n_files = 0

    for batch in tqdm(test_dataloader):
        for wav_path in batch:

            file_name_prefix = wav_path.split('/')[-1].split('.')[0]

            if convert:
                save_path = wav_path.split('.')[0]+'.wav'

                if not os.path.exists(wav_path) or wav_path.split('.')[-1] == 'wav':
                    pass
                else:
                    signal = AudioSignal(wav_path)
                    signal.write(save_path)

                    os.remove(wav_path)

                wav_path = save_path

            # DAC
            encoded_dac, meta_dac, audio_tensor = tokenize(wav_path, audio_secs, dac)
            #logging.info(f"encoded_dac.shape: {encoded_dac.shape} | audio_tensor.shape: {audio_tensor.shape}")
            total_generated_samples = encoded_dac.shape[-1] * dac.model.hop_length
            meta_dac['original_length'] = min(meta_dac['original_length'], total_generated_samples)
            decoded_dac = detokenize(encoded_dac, meta_dac, dac)

            # save
            save_path = os.path.join(dac_folder, file_name_prefix+'.wav')
            decoded_dac.write(save_path)

            smallest_last_dim = min(decoded_dac.shape[-1], audio_tensor.shape[-1])
            #logging.info(f"decoded_dac.shape: {decoded_dac.shape} | audio_tensor.shape: {audio_tensor.shape}\n")

            decoded_dac = decoded_dac.audio_data.squeeze()[:smallest_last_dim].cuda()
            audio_tensor = audio_tensor.squeeze()[:smallest_last_dim]

            dac_error += (decoded_dac - audio_tensor).abs().mean().item()

            # Encodec
            encoded_encodec, meta_encodec, audio_tensor = tokenize(wav_path, audio_secs, encodec)
            decoded_encodec = detokenize(encoded_encodec, meta_encodec, encodec)

            # save
            save_path = os.path.join(encodec_folder, file_name_prefix+'.wav')
            save_audio(save_path, decoded_encodec.squeeze().cpu(), sample_rate=encodec.sample_rate)

            smallest_last_dim = min(decoded_encodec.shape[-1], audio_tensor.shape[-1])
            #logging.info(f"decoded_encodec.shape: {decoded_encodec.shape} | audio_tensor.shape: {audio_tensor.shape}")

            decoded_encodec = decoded_encodec.squeeze()[:smallest_last_dim].cuda()
            audio_tensor = audio_tensor.squeeze()[:smallest_last_dim].cuda()

            encodec_error += (decoded_encodec - audio_tensor).abs().mean().item()

            n_files += 1

    dac_mae = dac_error / n_files
    encodec_mae = encodec_error / n_files
    logger.info(f"dac_mae: {dac_mae} | encodec_mae: {encodec_mae}")


if __name__ == "__main__":
    test_audio_tokenizers(DATASET_ROOT)