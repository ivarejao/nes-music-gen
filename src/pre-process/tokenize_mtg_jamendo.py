import os
import logging
import random
import csv

from tqdm import tqdm

import pandas as pd

import torch

from audiotools import AudioSignal

from any2music.audio.decoders.musicgen import DelayProvider
from any2music.text.encoders import T5Conditioner
from any2music.audio.tokenizers import DACCompressionModel
from any2music.audio.decoders.musicgen import MusicGenSize, MUSICGEN_SIZES
from any2music.audio.dataloaders.mtg_jamendo import MTGJamendoDataset


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_ROOT = "/home/es119256/dados/datasets/nes_dataset"

def select_samples(n, split, path):
    random.seed(42)

    dataset = MTGJamendoDataset(
        dataset_root='/home/es119256/dados/datasets/mtg-jamendo',
        mtg_clone='/home/es119256/dados/repos/passt-on-mtg-jamendo-dataset',
        split_type=split
    )

    split_csv = [['path', 'duration', 'desc']]

    range_dset = range(len(dataset))

    if n > 0:
        if n > len(dataset):
            raise ValueError("n must be less or equal to dataset size")

        idx_choices = random.choices(range_dset, k=n)
    else:    
        idx_choices = range_dset

    for idx in idx_choices:
        example = dataset[idx]
        split_csv.append(example.values())

    with open(path, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(split_csv)

    return path


def tokenize_audio(file_path_wav:str, audio_secs:int, audio_tokenizer):
    max_audio_len = audio_tokenizer.sample_rate*audio_secs
    min_audio_len = audio_tokenizer.sample_rate*8 # at least one bar in 4/4 120 bpm as in https://arxiv.org/pdf/2404.04420
    max_audio_tokens = int(audio_tokenizer.frame_rate*audio_secs)

    # DAC compress function already resamples
    audio_tensor = AudioSignal(file_path_wav, device='cuda').to_mono().cuda()

    logger.debug(f"\n\nTokenizing size {audio_tensor.shape[-1]} from {file_path_wav}")

    count = 1
    current_chunk = audio_tensor[:, :, :max_audio_len]

    while current_chunk.shape[-1] >= min_audio_len:
        logger.debug(f"\t Chunck {count} of size {current_chunk.shape[-1]}")
        with torch.no_grad():
            # Tokenize the audio
            encoded_audio, meta = audio_tokenizer.encode(current_chunk)
            encoded_audio = encoded_audio[:, :, :max_audio_tokens]

            # Add special tokens
            B, K, S = encoded_audio.shape

            bos = torch.full((B, K, 1), audio_tokenizer.bos_token_id, dtype=torch.long, device='cuda')
            eos = torch.full((B, K, 1), audio_tokenizer.eos_token_id, dtype=torch.long, device='cuda')
            padding = torch.full((B, K, 1), audio_tokenizer.pad_token_id, dtype=torch.long, device='cuda')

            delta = max_audio_tokens - S # If deta is a positive num, we get the value to get up to max_audio_tokens + min_padding, otherwise delta is 0
            bos_padding = 0 if count == 1 else 1 # we only add padding to account for bos from the second chunk onwards
            eos_padding = 1 if delta == 0 else 0 # we only add padding to account for eos when we have a chunk seq the same size as max_audio_tokens
            delay_padding = audio_tokenizer.num_codebooks-1 #  num_codebooks -1 for the delay pattern
            padding_size = delay_padding + delta + bos_padding + eos_padding

            if count == 1:
                encoded_audio = torch.cat([bos, encoded_audio], dim=-1)

            if delta > 0:
                encoded_audio = torch.cat([encoded_audio, eos], dim=-1)

            final_padding = padding.expand((-1, -1, padding_size))
            encoded_audio = torch.cat([encoded_audio, final_padding], dim=-1)

            logger.debug(f"\t\tdelta {delta} |  bos_padding {bos_padding} | eos_padding {eos_padding}")

            assert encoded_audio.shape[-1] == max_audio_tokens + delay_padding + 2

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

            yield input_tokens, target_tokens, meta, count

            current_chunk = audio_tensor[:, :, max_audio_len*count:max_audio_len*(count+1)]
            count += 1


def tokenize(
        csv_path:str,
        save_folder:str,
        decoder_size:MusicGenSize = MusicGenSize.SMALL,
        audio_secs:int=60
    ):
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder, exist_ok=True)

    size_params = MUSICGEN_SIZES[decoder_size.value]

    # Get df
    df = pd.read_csv(csv_path)

    # Get models
    dac = DACCompressionModel.get_pretrained("44khz", device='cuda').eval().cuda()
    dac.set_num_codebooks(4)

    t5 = T5Conditioner('t5-base', size_params.d_model, device='cuda').eval().cuda()

    for row in tqdm(df.to_dict(orient='records'), total=len(df), desc=csv_path):
        wav_path = row["path"]
        desc = row["desc"]

        # Tokenize audio
        for input_tokens, target_tokens, meta, count in tokenize_audio(wav_path, audio_secs, dac):
            file_name_prefix  = wav_path.split('/')[-1][:-4]

            if count > 1:
                file_name_prefix += f"_{count}"

            save_path = os.path.join(save_folder, file_name_prefix + '_pre_comp.pt')

            if os.path.isfile(save_path):
                continue

            # Get text tokens
            desc_tags = desc.split(',')
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


if __name__ == '__main__':
    # Get splits with the same size set by n. Negative values get the full dataset
    train_csv = select_samples(n=-1, split='train', path='./mtg_jamendo_train_full.csv')
    test_csv = select_samples(n=-1, split='test', path='./mtg_jamendo_test_full.csv')
    validation_csv = select_samples(n=-1, split='validation', path='./mtg_jamendo_validation_full.csv')

    # Tokenize splits data
    tokenize(train_csv, save_folder='/home/es119256/dados/datasets/tokenized_mtg_jamendo/train')
    tokenize(test_csv, save_folder='/home/es119256/dados/datasets/tokenized_mtg_jamendo/test')
    tokenize(validation_csv, save_folder='/home/es119256/dados/datasets/tokenized_mtg_jamendo/validation')
