import os
import random
import logging

from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import DataLoader

from any2music.audio.tokenizers import DACCompressionModel
from any2music.audio.decoders.musicgen import MusicGenTransformer, MusicGenSize

from src.dataset.snes_datasets import NESAudioDataset
# from src.dataset.mtg_jamendo_dataset import MTGJamendoAudioDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_ROOT_NES = "/home/es119256/dados/datasets/nes_dataset"
DATASET_ROOT_JAMENDO = '/home/es119256/dados/datasets/tokenized_mtg_jamendo'
XP_NAME = os.environ.get("XP_NAME", "xp_audio_dset_v0dot8")

# temperature -> step 0,25 de 1 até 2
# cfg_scale -> step 0,25 de 1,5 até 3
# top_k -> MIDI/NLM 10 até 100 step 10 | Áudio 50 até 250 step 50
# top_p -> 0.7 até 1 step 0.1 
TEMP_STEPS = np.arange(0.5, 2, 0.5)
CFG_STEPS = np.arange(1, 3, 0.5)
TOPK_STEPS = np.arange(50, 250, 50)
# TOPP_STEPS = np.arange(0.7, 0.9, 0.1)
TOPP_STEPS = [0.7]
STEPS = [(temp, cfg, top_k, top_p) for temp in TEMP_STEPS for cfg in CFG_STEPS for top_k in TOPK_STEPS for top_p in TOPP_STEPS]

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)


def collate(batch):
    input_tokens_batch = []
    target_tokens_batch = []
    meta_batch = []
    txt_tokens_batch = []
    txt_tokens_mask_batch = []
    file_name_prefix_batch = []

    for example in batch:
        input_tokens_batch.append(example['input_tokens'])
        target_tokens_batch.append(example['target_tokens'])
        meta_batch.append(example['meta'])
        txt_tokens_batch.append(example['txt_tokens'])
        txt_tokens_mask_batch.append(example['txt_tokens_mask'])
        file_name_prefix_batch.append(example['file_name_prefix'])

    input_tokens_batch = torch.stack(input_tokens_batch)
    target_tokens_batch = torch.stack(target_tokens_batch)
    txt_tokens_batch = torch.stack(txt_tokens_batch)
    txt_tokens_mask_batch = torch.stack(txt_tokens_mask_batch)

    return file_name_prefix_batch, input_tokens_batch, target_tokens_batch, meta_batch, txt_tokens_batch, txt_tokens_mask_batch


def run_generation(test_dataloader, model, audio_tokenizer, generation_folder, temperature, cfg, top_k, top_p):
    model.eval()

    iter_loader = iter(test_dataloader)
    batch = next(iter_loader)
    file_name_prefix, _, _, metas, txt_tokens, txt_tokens_mask = batch

    conditioners = txt_tokens.to(f"cuda")
    conditioner_mask = txt_tokens_mask.to(f"cuda")

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        tqdm.write(f"Generating audio tokens for {file_name_prefix}...")

        audio_tokens, valid_lengths = model.generate(
            max_new_tokens=model.max_seq_len,
            src=conditioners,
            src_mask=conditioner_mask,
            temperature=temperature,
            cfg_scale=cfg,
            top_k=top_k,
            top_p=top_p
        )

        B = conditioners.shape[0]
        for b in range(B):
            tqdm.write(f"(run_generation) Generating audio tokens for {file_name_prefix[b]}...")

            meta = metas[b]
            audio_tokens_b = audio_tokens[b].clone().unsqueeze(0)
            audio_tokens_b = torch.clamp(audio_tokens_b, min=0, max=audio_tokenizer.orig_vocab_size - 1)

            valid_samples = valid_lengths[b].item()
            audio_tokens_b = audio_tokens_b[:, :, :valid_samples]
            print(f"(run_generation_test) Clean audio codes shape: {audio_tokens_b.shape} | "
                f"valid_samples {valid_samples}")

            total_generated_samples = audio_tokens_b.shape[-1] * audio_tokenizer.model.hop_length
            meta["original_length"] = min(meta["original_length"], total_generated_samples)
            decoded_audio = audio_tokenizer.decode(audio_tokens_b.cpu(), meta)

            print(f"Decoded audio shape: {decoded_audio.shape}\n")
            decoded_audio.write(os.path.join(generation_folder, f"{file_name_prefix[b]}.wav"))

    model.train()


def pre_comp_grid_search(audio_secs, dataset_class, dataset_root, decoder_size, search_dir, checkpoint_path):
    # Get models
    dac = DACCompressionModel.get_pretrained("44khz", device='cuda').eval()
    dac.set_num_codebooks(4)

    model = MusicGenTransformer(
        vocab_size = dac.vocab_size,
        pad_token_id = dac.pad_token_id,
        eos_token_id = dac.eos_token_id,
        bos_token_id = dac.bos_token_id,
        frame_rate = int(dac.frame_rate),
        audio_duration = audio_secs,
        conditioner_d_model=768,
        model_size = decoder_size,
        dtype=torch.float32
    ).cuda().eval()

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(ckpt['model'], strict=True)
    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    # Get dataloader
    test_dataset = dataset_class(
        dataset_root=dataset_root, 
        split="test",
        num_codebooks=model.num_codebooks,
        audio_secs=audio_secs
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate,
    )

    for step in tqdm(STEPS):
        temp, cfg, top_k, top_p = step

        step_generation_folder = os.path.join(search_dir, f'temp_{temp}_cfg_{cfg}_topk_{top_k}_top_p_{top_p}')
        if not os.path.isdir(step_generation_folder):
            os.makedirs(step_generation_folder)

        run_generation(test_dataloader, model, dac, step_generation_folder, temp, cfg, top_k, top_p)


if __name__ == "__main__":
    set_seed()

    pre_comp_grid_search(
        audio_secs=60,
        dataset_class=NESAudioDataset,
        dataset_root=DATASET_ROOT_NES,
        decoder_size=MusicGenSize.SMALL,
        search_dir='/home/es119256/dados/repos/nes-music-gen/grid_search/audio',
        checkpoint_path='/home/es119256/dados/repos/nes-music-gen/xp_audio_PRE_TRAIN_dset_v0dot8/checkpoints/checkpoint_best.pth'
    )