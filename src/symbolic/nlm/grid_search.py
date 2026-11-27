import os
import json
import random
import logging

from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import DataLoader

from any2music.text.decoders.textgen import TextGenTransformer, TextGenSize

from src.dataset.snes_datasets import NESNlmDataset

PADDING_TKN = 0
BOS_TKN = 1
EOS_TKN = 2

DATASET_ROOT = "/home/es119256/dados/datasets/nes_dataset"

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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collate(batch):
    input_tokens_batch = []
    target_tokens_batch = []
    txt_tokens_batch = []
    txt_tokens_mask_batch = []
    file_name_prefix_batch = []

    max_seq_len = 0
    for example in batch:
        seq_len = example['target_tokens'].shape[-1]

        if seq_len > max_seq_len:
            max_seq_len = seq_len

    for example in batch:
        # pad input tokens
        input_tokens:torch.Tensor = example['input_tokens']
        pad_size = max_seq_len - input_tokens.shape[-1]
        padding = torch.zeros(pad_size, dtype=torch.long)
        input_tokens = torch.cat([input_tokens, padding])

        input_tokens_batch.append(input_tokens)

        # Pad target tokens
        target_tokens = example['target_tokens']
        padding = torch.zeros(pad_size, dtype=torch.long)
        target_tokens = torch.cat([target_tokens, padding])

        target_tokens_batch.append(target_tokens)

        # Other batch info
        txt_tokens_batch.append(example['txt_tokens'])
        txt_tokens_mask_batch.append(example['txt_tokens_mask'])
        file_name_prefix_batch.append(example['file_name_prefix'])

    input_tokens_batch = torch.stack(input_tokens_batch)
    target_tokens_batch = torch.stack(target_tokens_batch)


    txt_tokens_batch = torch.stack(txt_tokens_batch)
    txt_tokens_mask_batch = torch.stack(txt_tokens_mask_batch)

    return file_name_prefix_batch, input_tokens_batch, target_tokens_batch, txt_tokens_batch, txt_tokens_mask_batch


def run_generation(test_dataloader, model, step_generation_folder, temp, cfg, top_k, top_p):
    model.eval()
    iter_loader = iter(test_dataloader)
    batch = next(iter_loader)
    file_name_prefix, _, _, txt_tokens, txt_tokens_mask = batch

    logger.info(f"Generating audio tokens for {file_name_prefix}...")

    conditioners = txt_tokens.cuda()
    conditioner_mask = txt_tokens_mask.cuda()

    # Cast numpy to python
    temp = float(temp)
    cfg = float(cfg)
    top_k = int(top_k)
    top_p = float(top_p)

    # Run generation
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        tokens, valid_lengths = model.generate(
            max_new_tokens=model.max_seq_len,
            src=conditioners,
            src_mask=conditioner_mask,
            temperature=temp,
            cfg_scale=cfg,
            top_k=top_k,
            top_p=top_p
        )

        tokens = tokens.detach().cpu()
        print(f"Generated tokens of shape: {tokens.shape}\n")

        B = conditioners.shape[0]
        for b in range(B):
            print(f"(run_generation_test) Generating audio tokens for {file_name_prefix[b]}...")

            # Decode back to audio
            path = os.path.join(step_generation_folder, file_name_prefix[b]+'.json')

            tokens_b = tokens[b].clone()

            valid_samples = valid_lengths[b].item()
            tokens_b = tokens_b[:valid_samples]

            print(f"(run_generation_test) Tokens shape: {tokens_b.shape} | "
                              f"valid_samples {valid_samples}")

            with open(path, 'w') as f:
                json.dump(tokens_b.tolist(), f)


def pre_comp_grid_search(dataset_root, decoder_size, search_dir, checkpoint_path):
    # Get models
    model = TextGenTransformer(
        vocab_size = 565,
        pad_token_id = PADDING_TKN,
        bos_token_id = BOS_TKN,
        eos_token_id = EOS_TKN,
        max_seq_len = 6_000,
        conditioner_d_model=768,
        model_size = decoder_size,
        dtype=torch.float32
    ).cuda()

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    # Get dataloader
    test_dataset = NESNlmDataset(
        dataset_root=dataset_root, 
        split="test"
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate,
    )

    for step in tqdm(STEPS):
        temp, cfg, top_k, top_p = step

        step_generation_folder = os.path.join(search_dir, f'temp_{temp}_cfg_{cfg}_topk_{top_k}_top_p_{top_p}')
        if not os.path.isdir(step_generation_folder):
            os.makedirs(step_generation_folder)

        run_generation(test_dataloader, model, step_generation_folder, temp, cfg, top_k, top_p)


if __name__ == "__main__":
    set_seed()

    pre_comp_grid_search(
        dataset_root=DATASET_ROOT,
        decoder_size=TextGenSize.SMALL,
        search_dir='/home/es119256/dados/repos/nes-music-gen/grid_search/nlm',
        checkpoint_path='/home/es119256/dados/repos/nes-music-gen/xp_nlm_nes_dset_v0dot8/checkpoints/checkpoint_best.pth'
    )