import os
import logging

import torch
from torch.utils.data import DataLoader
from torchaudio import save as save_audio

from transformers import MusicgenProcessor, MusicgenForConditionalGeneration

from src.dataset.dataset import NESMusicGenBaseDataset


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_ROOT = "/home/es119256/dados/datasets/nes_dataset"

def get_xp(xp_dir):
    xp_folder = os.path.join(xp_dir, "my_xp_musicgen_base")
    generation_folder = os.path.join(xp_folder, "generation_tests")
    os.makedirs(generation_folder, exist_ok=True) # Ensure folder exists
    return xp_folder, generation_folder

def generate(
        dataset_root:str, 
        audio_secs:int=30,
        xp_dir = './'
    ):

    # Instantiate xp
    xp_folder, generation_folder = get_xp(xp_dir)

    # 1. Load the Processor and force Left Padding on BOTH sub-components
    processor = MusicgenProcessor.from_pretrained("facebook/musicgen-small")
    processor.tokenizer.padding_side = "left"
    if hasattr(processor, "padding_side"):
        processor.padding_side = "left"

    # 2. Load the model directly to CUDA
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to("cuda")
    
    tokenizer_fr:int = model.audio_encoder.config.frame_rate # type: ignore
    tokenizer_sr:int = model.config.audio_encoder.sampling_rate # type: ignore
    logger.info(f"tokenizer_fr: {tokenizer_fr} | tokenizer_sr: {tokenizer_sr}")

    # Get dataloader with batch_size=10
    test_dataset = NESMusicGenBaseDataset(
        dataset_root=dataset_root, 
        split="test"
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=10, # Kept at 10 as requested
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    for batch in test_dataloader:
        file_name_prefixes, desc_file_txts = batch

        # Run batch generation
        with torch.inference_mode():
            logger.info(f"Generating audio tokens for batch of {len(file_name_prefixes)} items...")
            
            # Pass the batch of strings directly
            inputs = processor(
                text=desc_file_txts,
                padding=True,
                return_tensors="pt",
            )
            # Send all processed tensors to GPU
            inputs = {k: v.to("cuda") for k, v in inputs.items() if isinstance(v, torch.Tensor)}

            # Generate all 10 tracks simultaneously
            audio_values = model.generate(
                **inputs, 
                do_sample=True, 
                guidance_scale=3, 
                max_new_tokens=audio_secs * tokenizer_fr
            )
            
            logger.info(f"Generated batch audio shape: {audio_values.shape}")

            # 3. Iterate through the batch items to save them individually
            for idx in range(audio_values.shape[0]):
                single_audio = audio_values[idx] # Shape: [channels, audio_samples]
                file_name = file_name_prefixes[idx]
                
                save_path = os.path.join(generation_folder, f"{file_name}.wav")
                
                # Squeeze is omitted; single_audio is already isolated perfectly for torchaudio
                save_audio(save_path, single_audio.cpu(), sample_rate=tokenizer_sr)

if __name__ == "__main__":
    generate(dataset_root=DATASET_ROOT)