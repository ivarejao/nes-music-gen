import os
import logging
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchaudio import save as save_audio

from audiotools import AudioSignal
from any2music.audio.tokenizers import DACCompressionModel, HFEncodecCompressionModel
from any2music.audio.utils import load_mono_and_resample


from src.dataset.dataset import NESTestTokenizersDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_ROOT = "/home/es119256/dados/datasets/nes_dataset"

def instantiate_xp(xp_dir):
    xp_folder = os.path.join(xp_dir, "test_audio_tokenizers")
    encodec_folder = os.path.join(xp_folder, "encodec")
    dac_folder = os.path.join(xp_folder, "dac")

    if not os.path.isdir(xp_folder):
        os.makedirs(xp_folder, exist_ok=True)

        for subfolder in [encodec_folder, dac_folder]:
            os.makedirs(subfolder, exist_ok=True)

    return xp_folder, encodec_folder, dac_folder


def tokenize(wav:str, audio_secs:int, audio_tokenizer):
    max_audio_len = audio_tokenizer.sample_rate*audio_secs
    max_audio_tokens = int(audio_tokenizer.frame_rate*audio_secs)

    if isinstance(wav, str):
        audio_tensor = AudioSignal(wav, device='cuda').to_mono()[:, :, :max_audio_len].cuda()
    else:
        audio_tensor = wav

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
        xp_dir = './'
    ):

    # Instanciate xp
    xp_folder, encodec_folder, dac_folder = instantiate_xp(xp_dir)

    # Get models
    dac = DACCompressionModel.get_pretrained("44khz", device='cpu').eval().cuda()
    dac.set_num_codebooks(4)

    encodec = HFEncodecCompressionModel.get_pretrained('facebook/encodec_32khz').cuda()

    # Get dataloader
    BATCH_SIZE = 12
    test_dataset = NESTestTokenizersDataset(
        dataset_root=dataset_root, 
        split="test"
    )
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
        file_name_prefix_batch, wav_path_batch = batch

        for file_name_prefix, wav_path in zip(file_name_prefix_batch, wav_path_batch):
            # DAC
            encoded_dac, meta_dac, audio_tensor = tokenize(wav_path, audio_secs, dac)
            decoded_dac = detokenize(encoded_dac, meta_dac, dac)
            smaller_last_dim = min(decoded_dac.shape[-1], audio_tensor.shape[-1])
            logging.info(f"{decoded_dac.shape} | {audio_tensor.shape}")
            dac_error += (decoded_dac.audio_data.squeeze()[:smaller_last_dim].cuda() - audio_tensor.squeeze()[:smaller_last_dim]).abs().mean().item()

            # Encodec
            audio_tensor = load_mono_and_resample(wav_path, encodec.sample_rate)[0]
            audio_tensor = audio_tensor.unsqueeze(0).cuda() # add batch dim

            encoded_encodec, meta_encodec, _ = tokenize(audio_tensor, audio_secs, encodec)
            decoded_encodec = detokenize(encoded_encodec, meta_encodec, encodec)
            smaller_last_dim = min(decoded_encodec.shape[-1], audio_tensor.shape[-1])
            logging.info(f"{decoded_encodec.shape} | {audio_tensor.shape} | smaller_last_dim {smaller_last_dim}")
            encodec_error += (decoded_encodec.squeeze()[:smaller_last_dim].cuda() - audio_tensor.squeeze()[:smaller_last_dim].cuda()).abs().mean().item()

            n_files += 1
    
    dac_mae = dac_error / n_files
    encodec_mae = encodec_error / n_files
    logger.info(f"dac_mae: {dac_mae} | encodec_mae: {encodec_mae}")


if __name__ == "__main__":
    test_audio_tokenizers(DATASET_ROOT)