import os
import math
import logging
import json
import random
from enum import Enum
from itertools import islice

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from any2music.text.decoders.textgen import TextGenTransformer, TextGenSize

PADDING_TKN = 0
BOS_TKN = 1
EOS_TKN = 2

class SYMBOLIC_TYPE(Enum):
    MIDI = "midi"
    NLM = "nlm"

logger = logging.getLogger(__name__)

def instantiate_xp(xp_dir, type:SYMBOLIC_TYPE):
    xp_folder = os.path.join(xp_dir, f"xp_{type.value}_nes_dset_v0dot8")
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

    iterator = tqdm(dataloader, total=len(dataloader), desc="Eval")

    with torch.inference_mode():
        for batch in iterator:
            _, input_tokens, target_tokens, txt_tokens, txt_tokens_mask = batch

            input_tokens = input_tokens.cuda()
            txt_tokens = txt_tokens.cuda()
            txt_tokens_mask = txt_tokens_mask.cuda()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(src=txt_tokens, tgt=input_tokens, src_mask=txt_tokens_mask)
                flat_logits = logits.reshape(-1, model.vocab_size)
                flat_targets = target_tokens.cuda().reshape(-1)

                loss = criterion(flat_logits, flat_targets)
                losses.append(loss.item())

    loss = sum(losses) / len(losses) if losses else float("nan")
    model.train()
    return loss


def run_generation_test(test_dataloader, model, generation_folder):
    model.eval()
    iter_loader = iter(test_dataloader)
    batch = next(iter_loader)
    file_name_prefix, _, _, txt_tokens, txt_tokens_mask = batch

    logger.info(f"Generating audio tokens for {file_name_prefix}...")

    conditioners = txt_tokens.cuda()
    conditioner_mask = txt_tokens_mask.cuda()

    # Run generation
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        tokens, valid_lengths = model.generate(
            max_new_tokens=model.max_seq_len,
            src=conditioners,
            src_mask=conditioner_mask,
            temperature=1,
            top_k=250,
            top_p=0.8
        )

        tokens = tokens.detach().cpu()
        print(f"Generated tokens of shape: {tokens.shape}\n")

        B = conditioners.shape[0]
        for b in range(B):
            print(f"(run_generation_test) Generating audio tokens for {file_name_prefix[b]}...")

            # Decode back to audio
            path = os.path.join(generation_folder, file_name_prefix[b]+'.json')

            tokens_b = tokens[b].clone()

            valid_samples = valid_lengths[b].item()
            tokens_b = tokens_b[:valid_samples]

            print(f"(run_generation_test) Tokens shape: {tokens_b.shape} | "
                              f"valid_samples {valid_samples}")

            with open(path, 'w') as f:
                json.dump(tokens_b.tolist(), f)

    model.train()


def train_symbolic(
        type:SYMBOLIC_TYPE,
        dataset_root:str, 
        epochs:int,
        early_stop_patience: int,
        warmup_steps:int,
        decoder_size:TextGenSize,
        generate_every: int,
        xp_dir = './'
    ):

    if type == SYMBOLIC_TYPE.MIDI:
        from src.dataset.snes_datasets import NESMidiDataset as NESSymbDataset
    elif type == SYMBOLIC_TYPE.NLM:
        from src.dataset.snes_datasets import NESNlmDataset as NESSymbDataset
    else:
        raise TypeError("Invalid train_symbolic type")

    # Instanciate xp
    xp_folder, checkpoint_folder, generation_folder = instantiate_xp(xp_dir, type)

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

    # Get dataloader
    train_dataset = NESSymbDataset(
        dataset_root=dataset_root, 
        split="train"
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate,
    )

    test_dataset = NESSymbDataset(
        dataset_root=dataset_root, 
        split="test"
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate,
    )

    # Train
    criterion = torch.nn.CrossEntropyLoss(ignore_index=model.pad_token_id) # Ignore the padding tokens in the loss calculation
    optim = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler =  get_cosine_with_warmup_scheduler(optim, warmup_steps, epochs*len(train_dataloader))

    best_test_loss = float("inf")
    train_losses = []
    test_losses = []
    current_epoch = 0
    epochs_wo_improvement = 0

    checkpoint_last_path = os.path.join(checkpoint_folder, f'checkpoint_last.pth')
    checkpoint_best_path = os.path.join(checkpoint_folder, f'checkpoint_best.pth')

    if os.path.isfile(checkpoint_last_path):
        ckpt = torch.load(checkpoint_last_path, map_location=f'cuda:0', weights_only=False)

        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        scheduler.load_state_dict(ckpt["scheduler"])

        best_test_loss = ckpt["best_test_loss"]
        train_losses = ckpt["loss_history"]["train"]
        test_losses = ckpt["loss_history"]["test"]
        current_epoch = ckpt["epoch"] + 1
        epochs_wo_improvement = ckpt["epochs_wo_improvement"]

        logger.info(f"Loaded checkpoint from {checkpoint_last_path} | Current Epoch: {current_epoch} | Best Test Loss {best_test_loss:.4f}")


    for epoch in range(current_epoch, epochs):
        if not model.training: model.train()

        epoch_losses = []
        iterator = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {epoch}")

        for update, batch in enumerate(iterator):
            _, input_tokens, target_tokens, txt_tokens, txt_tokens_mask = batch

            input_tokens = input_tokens.cuda().long()

            if random.random() < 0.3:
                drop_conditioning = True
                txt_tokens = None
                txt_tokens_mask = None
            else:
                drop_conditioning = False
                txt_tokens = txt_tokens.cuda()
                txt_tokens_mask = txt_tokens_mask.cuda()

            optim.zero_grad()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(src=txt_tokens, src_mask=txt_tokens_mask, tgt=input_tokens, drop_conditioning=drop_conditioning)

                # Reshape for CrossEntropyLoss
                # Logits: (B, S, Vocab) -> (B * S, Vocab)
                # Targets: (B, S) -> (B * S)
                flat_logits = logits.reshape(-1, model.vocab_size)
                flat_targets = target_tokens.cuda().long().reshape(-1)

                # Compute loss and step
                loss = criterion(flat_logits, flat_targets)
                loss_item = loss.item()
                epoch_losses.append(loss_item)

            loss.backward()

            # Clip gradients to a max norm of 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optim.step()
            scheduler.step()

        epoch_loss = sum(epoch_losses) / len(epoch_losses)
        train_losses.append(epoch_loss)

        test_loss = evaluate_model_loss(model, test_dataloader, criterion)
        test_losses.append(test_loss)
        save_loss_history(xp_folder, train_losses, test_losses)

        is_best = test_loss <= best_test_loss
        if is_best:
            best_test_loss = test_loss
            epochs_wo_improvement = 0

        logger.info(f"Epoch {epoch} Train Loss: {epoch_loss:.4f} | Test Loss: {test_loss:.4f} | Best Test  Loss: {best_test_loss:.4f}")

        if not is_best:
            epochs_wo_improvement += 1
            logger.info(f"------> {epochs_wo_improvement} epochs without improvement. Patience is {early_stop_patience}")

        ckpt = {
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "epochs_wo_improvement": epochs_wo_improvement,
            "best_test_loss": best_test_loss,
            "loss_history": {
                "train": train_losses,
                "test": test_losses
            }
        }
        torch.save(ckpt, checkpoint_last_path)

        if is_best:
            logger.info(f"Saved BEST checkpoint to {checkpoint_best_path} | Current Epoch: {epoch} | Best Test Loss {test_loss:.4f}")
            ckpt["best_test_loss"] = best_test_loss
            torch.save(ckpt, checkpoint_best_path)

        if generate_every > 0 and epoch % generate_every == 0:
            gen_epoch_folder = os.path.join(generation_folder, f"epoch_{epoch}")
            os.makedirs(gen_epoch_folder, exist_ok=True)
            run_generation_test(test_dataloader, model, gen_epoch_folder)

        if (early_stop_patience > 0) and (epochs_wo_improvement >= early_stop_patience):
            logger.info(f"######################## Early Stoping ########################")
            break
