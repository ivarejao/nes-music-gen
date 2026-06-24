import os
import math
import logging
import json
from functools import partial

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from audiotools import AudioSignal

from any2music.audio.decoders.musicgen import DelayProvider
from any2music.text.encoders import T5Conditioner
from any2music.audio.tokenizers import DACCompressionModel
from any2music.audio.decoders.musicgen import MusicGenTransformer, MusicGenSize

from src.dataset.dataset import NESDataset

logger = logging.getLogger(__name__)

def instantiate_xp(xp_dir):
    xp_folder = os.path.join(xp_dir, "my_xp")
    checkpoint_folder = os.path.join(xp_folder, "checkpoints")
    generation_folder = os.path.join(xp_folder, "generation_tests")

    if not os.path.isdir(xp_folder):
        os.makedirs(xp_folder, exist_ok=True)

        for subfolder in [checkpoint_folder, generation_folder]:
            os.makedirs(subfolder, exist_ok=True)

    return xp_folder, checkpoint_folder, generation_folder


def get_cosine_with_warmup_scheduler(optimizer, num_warmup_steps, num_training_steps, eta_min_ratio=0.1):
    """
    Creates a learning rate scheduler that increases linearly from 0 to the base LR 
    during warmup, then decreases following a cosine curve to a fraction of the base LR.
    """
    def lr_lambda(current_step):
        # 1. Linear Warmup Phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # 2. Cosine Decay Phase
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        progress = min(1.0, max(0.0, progress)) # Clip bounds
        
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        # Scale decay so it stops at eta_min_ratio (e.g., 10% of max LR) instead of 0
        return eta_min_ratio + (1.0 - eta_min_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def tokenize_audio(file_path_wav:str, audio_secs:int, audio_tokenizer):
    max_audio_len = audio_tokenizer.sample_rate*audio_secs
    max_audio_tokens = int(audio_tokenizer.frame_rate*audio_secs)

    # Keep preprocessing on CPU so dataloader workers can safely serialize results.
    audio_tensor = AudioSignal(file_path_wav, device='cpu').to_mono()[:, :, :max_audio_len]

    # Tokenize the audio
    with torch.inference_mode():
        encoded_audio, meta = audio_tokenizer.encode(audio_tensor)

        # Add padding so we don't loose the first token for the delay
        B, K, S = encoded_audio.shape
        padding = torch.full((B, K, 1), audio_tokenizer.pad_token_id, dtype=torch.long)
        # Add 
        if S < max_audio_tokens:
            final_padding = padding.expand((-1, -1, max_audio_tokens -( S + 1))) # +1 for the padding token
            encoded_audio = torch.cat([padding, encoded_audio, final_padding], dim=-1)
        else:
            encoded_audio = torch.cat([padding, encoded_audio], dim=-1)[:, :, :max_audio_tokens]

        # Apply the delay pattern for MusicGen
        # This shifts codebook 1 by 0, codebook 2 by 1, codebook 3 by 2, etc.
        delayed_audio, _ = DelayProvider.build_delay_pattern_mask(
            input_ids=encoded_audio,
            pad_token_id=audio_tokenizer.pad_token_id,
            max_length=encoded_audio.shape[-1] + audio_tokenizer.num_codebooks,
            audio_channels=1
        )

        input_tokens = delayed_audio[:, :, :-1].detach()
        target_tokens = delayed_audio[:, :, 1:].detach()

    return input_tokens, target_tokens, meta


def collate_fn(batch, text_tokenizer, audio_secs, audio_tokenizer):
    input_tokens_batch = []
    target_tokens_batch = []
    meta_batch = []
    description_batch = []
    file_name_prefix_batch = []

    for example in batch:
        input_tokens, target_tokens, meta = tokenize_audio(example['wav_path'], audio_secs, audio_tokenizer)
        input_tokens_batch.append(input_tokens)
        target_tokens_batch.append(target_tokens)
        meta_batch.append(meta)

        description_batch.append(example['description'])

        file_name_prefix_batch.append(example['file_name_prefix'])

    input_tokens_batch = torch.cat(input_tokens_batch, dim=0)
    target_tokens_batch = torch.cat(target_tokens_batch, dim=0)

    txt_tokens_batch = text_tokenizer.tokenize(description_batch)
    txt_tokens_batch, _ = text_tokenizer(txt_tokens_batch)
    txt_tokens_batch = txt_tokens_batch.to(torch.bfloat16).detach()

    return file_name_prefix_batch, input_tokens_batch, target_tokens_batch, meta_batch, txt_tokens_batch


def save_loss_history(xp_folder, train_losses, test_losses):
    loss_path = os.path.join(xp_folder, "losses.json")
    payload = {
        "train_loss": train_losses,
        "test_loss": test_losses,
    }

    with open(loss_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    logger.info(f"Saved train/test loss history to {loss_path}")


def evaluate_model_loss(model, dataloader, criterion):
    model.eval()
    losses = []

    with torch.inference_mode():
        for batch in dataloader:
            _, input_tokens, target_tokens, _, txt_tokens = batch

            input_tokens = input_tokens.cuda()
            txt_tokens = txt_tokens.cuda()

            logits = model(src=txt_tokens, tgt=input_tokens)
            flat_logits = logits.reshape(-1, model.vocab_size)
            flat_targets = target_tokens.cuda().reshape(-1)

            loss = criterion(flat_logits, flat_targets)
            losses.append(loss.item())

    loss = sum(losses) / len(losses) if losses else float("nan")
    model.train()
    return loss


def train_audio(
        dataset_root:str, 
        epochs:int, 
        warmup_steps:int, 
        audio_length:int,
        decoder_size:MusicGenSize,
        checkpoint_every:int = 5,
        generate_amount:int = 5,
        audio_secs:int=60,
        xp_dir = './'
    ):

    # Instanciate xp
    xp_folder, checkpoint_folder, generation_folder = instantiate_xp(xp_dir)

    # Get models
    dac = DACCompressionModel.get_pretrained("44khz", device='cpu').eval()
    dac.set_num_codebooks(4)

    model = MusicGenTransformer(
        vocab_size = dac.vocab_size,
        frame_rate = int(dac.frame_rate),
        audio_duration = audio_length,
        pad_token_id = dac.pad_token_id,
        eos_token_id = dac.eos_token_id,
        model_size = decoder_size
    ).cuda()

    t5 = T5Conditioner('t5-base', model.size_params.d_model, device='cpu').eval()

    # Get dataloader
    collate = partial(
        collate_fn, 
        text_tokenizer=t5, 
        audio_secs=audio_secs, 
        audio_tokenizer=dac
    )
    
    train_dataset = NESDataset(
        dataset_root=dataset_root, 
        split="train",
        num_codebooks=model.num_codebooks,
        audio_secs=audio_secs
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=12,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate,
    )

    test_dataset = NESDataset(
        dataset_root=dataset_root, 
        split="test",
        num_codebooks=model.num_codebooks,
        audio_secs=audio_secs
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=12,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate,
    )

    # Train
    criterium = torch.nn.CrossEntropyLoss(ignore_index=model.pad_token_id) # Ignore the padding tokens in the loss calculation
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler =  get_cosine_with_warmup_scheduler(optim, warmup_steps, epochs*len(train_dataloader))

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        if not model.training: model.train()

        epoch_loss = []
        for update, batch in enumerate(train_dataloader):
            _, input_tokens, target_tokens, _, txt_tokens = batch

            input_tokens = input_tokens.cuda()
            txt_tokens = txt_tokens.cuda()
    
            optim.zero_grad()
            logits = model(src=txt_tokens, tgt=input_tokens)

            # Reshape for CrossEntropyLoss
            # Logits: (B, K, S, Vocab) -> (B * K * S, Vocab)
            # Targets: (B, K, S) -> (B * K * S)
            flat_logits = logits.reshape(-1, model.vocab_size)
            flat_targets = target_tokens.cuda().reshape(-1)

            # Compute loss and step
            loss = criterium(flat_logits, flat_targets)
            loss_item = loss.item()
            logger.info(f"Epoch {epoch} -> Update {update} | Loss: {loss_item:.4f}")
            epoch_loss.append(loss_item)

            loss.backward()
            optim.step()
            scheduler.step()

        epoch_loss = torch.stack(epoch_loss).mean().item()
        train_losses.append(epoch_loss)

        test_loss = evaluate_model_loss(model, test_dataloader, criterium)
        test_losses.append(test_loss)
        save_loss_history(xp_folder, train_losses, test_losses)

        logger.info(f"Epoch {epoch} Train Loss: {epoch_loss:.4f} | Test Loss: {test_loss:.4f}")

        if epoch % checkpoint_every == 0:
            run_generation_test(generate_amount, test_dataloader, model, dac, generation_folder)

        ckpt_path = os.path.join(checkpoint_folder, f'checkpoint_{epoch}.pth')
        model.save(ckpt_path)


def run_generation_test(generate_amount, test_dataloader, model, audio_tokenizer, generation_folder):
    model.eval()
    iter_loader = iter(test_dataloader)

    for _ in range(generate_amount):
        batch = next(iter_loader)
        file_name_prefix, _, _, meta, txt_tokens = batch

        txt_tokens = txt_tokens.cuda()

        # Run generation
        with torch.inference_mode():
            logger.info(f"Generating audio tokens for {file_name_prefix}...")
            audio_tokens = model.generate(
                src=txt_tokens,
                temperature=1e-4, # < 1 -> eliminate randomness | = 1 -> the distribution learned | > 1 -> aproximate a uniform distribution
                top_k=1
            )

            print(f"Generate audio codes shape: {audio_tokens.shape}\n")

            # Decode back to audio
            total_generated_samples = audio_tokens.shape[-1] * audio_tokenizer.model.hop_length
            meta['original_length'] = min(meta['original_length'], total_generated_samples)
            decoded_audio = audio_tokenizer.decode(audio_tokens.cpu(), meta)

            logger.info(f"Decoded audio shape: {decoded_audio.shape}")
            path = os.path.join(generation_folder, file_name_prefix+'.wav')
            decoded_audio.write(path)

    model.train()