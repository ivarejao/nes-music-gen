import os
import logging

import torch
from torch.utils.data import DataLoader

from any2music.audio.tokenizers import DACCompressionModel
from any2music.audio.decoders.musicgen import MusicGenTransformer, MusicGenSize

from src.dataset.snes_datasets import NESAudioDataset
from src.dataset.mtg_jamendo_dataset import MTGJamendoAudioDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_ROOT_SNES = "/home/es119256/dados/datasets/nes_dataset"
DATASET_ROOT_JAMENDO = '/home/es119256/dados/datasets/tokenized_mtg_jamendo'

def get_xp(xp_dir):
    xp_folder = os.path.join(xp_dir, "my_xp_cfg_mtg_jamendo")
    checkpoint_folder = os.path.join(xp_folder, "checkpoints")
    generation_folder = os.path.join(xp_folder, "generation_tests")

    return xp_folder, checkpoint_folder, generation_folder

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

def generate(
        dataset_class,
        dataset_root:str, 
        epoch:int, 
        decoder_size:MusicGenSize,
        audio_secs:int=60,
        xp_dir = './'
    ):

    # Instanciate xp
    xp_folder, checkpoint_folder, generation_folder = get_xp(xp_dir)
    checkpoint_path = os.path.join(checkpoint_folder, f'checkpoint_{epoch}.pth')

    # Get models
    dac = DACCompressionModel.get_pretrained("44khz", device='cpu').eval()
    dac.set_num_codebooks(4)

    model = MusicGenTransformer(
        vocab_size = dac.vocab_size,
        frame_rate = int(dac.frame_rate),
        audio_duration = audio_secs,
        pad_token_id = dac.pad_token_id,
        eos_token_id = dac.eos_token_id,
        bos_token_id = dac.bos_token_id,
        model_size = decoder_size
    ).cuda().eval()


    logger.info(f"Loading checkpoint from {checkpoint_path}")
    full_loaded_object = torch.load(checkpoint_path, weights_only=False)

    if hasattr(full_loaded_object, 'state_dict'):
        ckpt = full_loaded_object.state_dict()
    else:
        ckpt = full_loaded_object

    model.load_state_dict(ckpt)
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
        batch_size=10,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate,
    )

    for batch in test_dataloader:
        file_name_prefix, input_tokens, target_tokens, meta, txt_tokens, txt_tokens_mask = batch

        for i in range(len(file_name_prefix)):
            conditioner = txt_tokens[i].unsqueeze(0).cuda()
            conditioner_mask = txt_tokens_mask[i].unsqueeze(0).cuda()

            # Run generation
            with torch.inference_mode():
                logger.info(f"Generating audio tokens for {file_name_prefix[i]}...")
                audio_tokens = model.generate(
                    max_new_tokens=model.max_seq_len,
                    src=conditioner,
                    src_mask=conditioner_mask,
                    temperature=1, # < 1 -> eliminate randomness | = 1 -> the distribution learned | > 1 -> aproximate a uniform distribution
                    cfg_scale=3.0,
                    top_k=10
                )

                print(f"Generate audio codes shape: {audio_tokens.shape}\n")

                # Decode back to audio
                audio_tokens = torch.clamp(audio_tokens, min=0, max=dac.orig_vocab_size - 1)
                total_generated_samples = audio_tokens.shape[-1] * dac.model.hop_length
                meta[i]['original_length'] = min(meta[i]['original_length'], total_generated_samples)
                decoded_audio = dac.decode(audio_tokens.cpu(), meta[i])

                logger.info(f"Decoded audio shape: {decoded_audio.shape}")
                path = os.path.join(generation_folder, file_name_prefix[i]+'.wav')
                decoded_audio.write(path)


if __name__ == "__main__":
    generate(
        dataset_class=MTGJamendoAudioDataset,
        dataset_root=DATASET_ROOT_JAMENDO,
        epoch=50,
        decoder_size=MusicGenSize.SMALL
    )