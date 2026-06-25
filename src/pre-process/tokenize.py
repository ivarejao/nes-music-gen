import os
import logging
import random

import pandas as pd

import torch

from audiotools import AudioSignal

from any2music.audio.decoders.musicgen import DelayProvider
from any2music.text.encoders import T5Conditioner
from any2music.audio.tokenizers import DACCompressionModel
from any2music.audio.decoders.musicgen import MusicGenSize, MUSICGEN_SIZES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_ROOT = "/home/es119256/dados/datasets/nes_dataset"

def tokenize_audio(file_path_wav:str, audio_secs:int, audio_tokenizer):
    max_audio_len = audio_tokenizer.sample_rate*audio_secs
    max_audio_tokens = int(audio_tokenizer.frame_rate*audio_secs)

    # Keep preprocessing on CPU so dataloader workers can safely serialize results.
    audio_tensor = AudioSignal(file_path_wav, device='cuda').to_mono()[:, :, :max_audio_len].cuda()

    with torch.no_grad():
        # Tokenize the audio
        encoded_audio, meta = audio_tokenizer.encode(audio_tensor)
        encoded_audio = encoded_audio[:, :, :max_audio_tokens]

        # Add special tokens
        B, K, S = encoded_audio.shape

        bos = torch.full((B, K, 1), audio_tokenizer.bos_token_id, dtype=torch.long, device='cuda')
        eos = torch.full((B, K, 1), audio_tokenizer.eos_token_id, dtype=torch.long, device='cuda')
        padding = torch.full((B, K, 1), audio_tokenizer.pad_token_id, dtype=torch.long, device='cuda')

        min_padding = audio_tokenizer.num_codebooks-1 #  num_codebooks -1 for the delay pattern
        delta = max_audio_tokens - S
        padding_size = delta + min_padding # If delta==0 we get min_padding. If deta is a positive num, we get the value to get up to max_audio_tokens + min_padding
        padding_size = max(min_padding, delta + min_padding)

        final_padding = padding.expand((-1, -1, padding_size)) 
        encoded_audio = torch.cat([bos, encoded_audio, eos, final_padding], dim=-1)

        assert encoded_audio.shape[-1] == max_audio_tokens + min_padding + 2

        # Apply the delay pattern for MusicGen
        # This shifts codebook 1 by 0, codebook 2 by 1, codebook 3 by 2, etc.
        delayed_audio, _ = DelayProvider.build_delay_pattern_mask(
            input_ids=encoded_audio,
            pad_token_id=audio_tokenizer.pad_token_id,
            max_length=encoded_audio.shape[-1] + audio_tokenizer.num_codebooks,
            audio_channels=1
        )

    input_tokens = delayed_audio[:, :, :-1].detach().cpu().contiguous().squeeze(0).clone()
    target_tokens = delayed_audio[:, :, 1:].detach().cpu().contiguous().squeeze(0).clone()

    return input_tokens, target_tokens, meta


def tokenize(
        dataset_root:str=DATASET_ROOT,
        decoder_size:MusicGenSize = MusicGenSize.SMALL,
        audio_secs:int=60
    ):

    csv_path = os.path.join(dataset_root, "nes_metadata.csv")
    size_params = MUSICGEN_SIZES[decoder_size.value]

    # Get clean df
    df = pd.read_csv(csv_path, dtype={"id":"object"})
    df = df[df.has_midi & df.has_nlm & df.has_music_track_description]
    df:pd.DataFrame = df.reset_index()

    # Get models
    dac = DACCompressionModel.get_pretrained("44khz", device='cuda').eval().cuda()
    dac.set_num_codebooks(4)

    t5 = T5Conditioner('t5-base', size_params.d_model, device='cuda').eval().cuda()

    for row in df.to_dict(orient='records'):
        file_name_prefix = row["file_name_prefix"]
        wav_path = os.path.join(dataset_root, row["file_path_wav"])
        desc_file_path = os.path.join(dataset_root, row["file_path_music_track_description"])

        save_path = wav_path[:-4] + '_pre_comp.pt'

        # Tokenize audio
        input_tokens, target_tokens, meta = tokenize_audio(wav_path, audio_secs, dac)

        # Get text tokens
        with open(desc_file_path, 'r') as f:
            file_txt = f.read()

        desc_tags = file_txt.split(',')
        random.shuffle(desc_tags)
        desc_tags = [tag.strip() for tag in desc_tags]
        desc_tags = ", ".join(desc_tags)
        logger.debug(f"Audio description: {desc_tags}")

        txt_tokens = t5.tokenize([desc_tags])
        txt_tokens, txt_tokens_mask = t5(txt_tokens)

        txt_tokens = txt_tokens.to(torch.bfloat16).cpu().contiguous().squeeze(0).detach()
        txt_tokens_mask = txt_tokens_mask.bool().cpu().contiguous().squeeze(0).detach()

        logger.debug(f"txt_tokens_mask {txt_tokens_mask.shape}\n{txt_tokens_mask}\n")

        save_dict = {
            'txt_tokens': txt_tokens,
            'txt_tokens_mask': txt_tokens_mask,
            'input_tokens': input_tokens,
            'target_tokens': target_tokens,
            'meta': meta
        }

        torch.save(save_dict, save_path)

        logger.info(f"Saved {file_name_prefix}")

if __name__ == '__main__':
    tokenize()

