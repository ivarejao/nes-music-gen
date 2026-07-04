import os
import json
import logging

import torch
from torch.utils.data import DataLoader

import any2music
from any2music.text.decoders.textgen import TextGenTransformer, TextGenSize

from src.dataset.dataset import NESMidiDataset

PADDING_TKN = 0
BOS_TKN = 1
EOS_TKN = 2


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_ROOT = "/home/es119256/dados/datasets/nes_dataset"

def get_xp(xp_dir):
    xp_folder = os.path.join(xp_dir, "my_xp_cfg_midi")
    checkpoint_folder = os.path.join(xp_folder, "checkpoints")
    generation_folder = os.path.join(xp_folder, "generation_tests")

    return xp_folder, checkpoint_folder, generation_folder

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


def generate(
        dataset_root:str, 
        epoch:int, 
        decoder_size:TextGenSize,
        xp_dir = './'
    ):

    # Instanciate xp
    xp_folder, checkpoint_folder, generation_folder = get_xp(xp_dir)
    checkpoint_path = os.path.join(checkpoint_folder, f'checkpoint_{epoch}.pth')

    # Get models
    model = TextGenTransformer(
        vocab_size = 565,
        pad_token_id = PADDING_TKN,
        bos_token_id = BOS_TKN,
        eos_token_id = EOS_TKN,
        max_seq_len = 10_000,
        model_size = decoder_size
    ).cuda()

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    full_loaded_object = torch.load(checkpoint_path, weights_only=False)

    if hasattr(full_loaded_object, 'state_dict'):
        ckpt = full_loaded_object.state_dict()
    else:
        ckpt = full_loaded_object

    model.load_state_dict(ckpt)
    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    # Get dataloader
    test_dataset = NESMidiDataset(
        dataset_root=dataset_root, 
        split="test"
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=10,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate,
    )

    for batch in test_dataloader:
        file_name_prefix, input_tokens, target_tokens, txt_tokens, txt_tokens_mask = batch

        for i in range(len(file_name_prefix)):
            conditioner = txt_tokens[i].unsqueeze(0).cuda()
            conditioner_mask = txt_tokens_mask[i].unsqueeze(0).cuda()

            # Run generation
            with torch.inference_mode():
                logger.info(f"Generating audio tokens for {file_name_prefix[i]}...")
                midi_tokens = model.generate(
                    max_new_tokens=model.max_seq_len,
                    src=conditioner,
                    src_mask=conditioner_mask,
                    temperature=1, # < 1 -> eliminate randomness | = 1 -> the distribution learned | > 1 -> aproximate a uniform distribution
                    cfg_scale=3.0,
                    top_k=10
                )

                logger.info(f"Decoded midi shape: {midi_tokens.shape}")
                path = os.path.join(generation_folder, file_name_prefix[i]+'.json')

                midi_tokens = midi_tokens.detach().cpu()
                with open(path, 'w') as f:
                    json.dump(midi_tokens.tolist(), f)


if __name__ == "__main__":
    generate(
        dataset_root=DATASET_ROOT,
        epoch=50,
        decoder_size=TextGenSize.SMALL
    )