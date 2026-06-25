import os
import math
import logging
import json
import random

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from any2music.audio.tokenizers import DACCompressionModel
from any2music.audio.decoders.musicgen import MusicGenTransformer, MusicGenSize

from src.dataset.dataset import NESDataset

logger = logging.getLogger(__name__)

def instantiate_xp(xp_dir):
    xp_folder = os.path.join(xp_dir, "my_xp_cfg")
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
            _, input_tokens, target_tokens, _, txt_tokens, txt_tokens_mask  = batch

            input_tokens = input_tokens.cuda()
            txt_tokens = txt_tokens.cuda()
            txt_tokens_mask = txt_tokens_mask.cuda()

            logits = model(src=txt_tokens, tgt=input_tokens, src_mask=txt_tokens_mask)
            flat_logits = logits.reshape(-1, model.vocab_size)
            flat_targets = target_tokens.cuda().reshape(-1)

            loss = criterion(flat_logits, flat_targets)
            losses.append(loss.item())

    loss = sum(losses) / len(losses) if losses else float("nan")
    model.train()
    return loss


def run_generation_test(test_dataloader, model, audio_tokenizer, generation_folder):
    model.eval()
    iter_loader = iter(test_dataloader)
    batch = next(iter_loader)
    file_name_prefix, _, _, meta, txt_tokens, txt_tokens_mask = batch

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
                top_k=250
            )

            print(f"Generate audio codes shape: {audio_tokens.shape}\n")

            # Decode back to audio
            total_generated_samples = audio_tokens.shape[-1] * audio_tokenizer.model.hop_length
            meta[i]['original_length'] = min(meta[i]['original_length'], total_generated_samples)
            decoded_audio = audio_tokenizer.decode(audio_tokens.cpu(), meta[i])

            logger.info(f"Decoded audio shape: {decoded_audio.shape}")
            path = os.path.join(generation_folder, file_name_prefix[i]+'.wav')
            decoded_audio.write(path)

    model.train()


def train_audio(
        dataset_root:str, 
        epochs:int, 
        warmup_steps:int,
        decoder_size:MusicGenSize,
        checkpoint_every:int = 10,
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
        audio_duration = audio_secs,
        pad_token_id = dac.pad_token_id,
        eos_token_id = dac.eos_token_id,
        bos_token_id = dac.bos_token_id,
        model_size = decoder_size
    ).cuda()

    # Get dataloader
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
    optim = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler =  get_cosine_with_warmup_scheduler(optim, warmup_steps, epochs*len(train_dataloader))

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        if not model.training: model.train()

        epoch_losses = []
        for update, batch in enumerate(train_dataloader):
            _, input_tokens, target_tokens, _, txt_tokens, txt_tokens_mask = batch

            input_tokens = input_tokens.cuda()

            if random.random() < 0.3:
                drop_conditioning = True
                txt_tokens = None
                txt_tokens_mask = None
            else:
                drop_conditioning = False
                txt_tokens = txt_tokens.cuda()
                txt_tokens_mask = txt_tokens_mask.cuda()

            optim.zero_grad()
            logits = model(src=txt_tokens, src_mask=txt_tokens_mask, tgt=input_tokens, drop_conditioning=drop_conditioning)

            # Reshape for CrossEntropyLoss
            # Logits: (B, K, S, Vocab) -> (B * K * S, Vocab)
            # Targets: (B, K, S) -> (B * K * S)
            flat_logits = logits.reshape(-1, model.vocab_size)
            flat_targets = target_tokens.cuda().reshape(-1)

            # Compute loss and step
            loss = criterium(flat_logits, flat_targets)
            loss_item = loss.item()
            logger.info(f"Epoch {epoch} -> Update {update} | Loss: {loss_item:.4f}")
            epoch_losses.append(loss_item)

            loss.backward()
            optim.step()
            scheduler.step()

        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        train_losses.append(epoch_loss)

        test_loss = evaluate_model_loss(model, test_dataloader, criterium)
        test_losses.append(test_loss)
        save_loss_history(xp_folder, train_losses, test_losses)

        logger.info(f"Epoch {epoch} Train Loss: {epoch_loss:.4f} | Test Loss: {test_loss:.4f}")

        if epoch % checkpoint_every == 0:
        #     run_generation_test(test_dataloader, model, dac, generation_folder)

            ckpt_path = os.path.join(checkpoint_folder, f'checkpoint_{epoch}.pth')
            torch.save(model.state_dict(), ckpt_path)