import os
import math
import logging
import json
import random
import signal
import sys

from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import LambdaLR

from any2music.audio.tokenizers import DACCompressionModel
from any2music.audio.decoders.musicgen import MusicGenTransformer, MusicGenSize


logger = logging.getLogger(__name__)

XP_NAME = "xp_jamendo_full_5e-4"

def instantiate_xp(xp_dir):
    xp_folder = os.path.join(xp_dir, XP_NAME)
    checkpoint_folder = os.path.join(xp_folder, "checkpoints")
    generation_folder = os.path.join(xp_folder, "generation_tests")

    if not os.path.isdir(xp_folder):
        os.makedirs(xp_folder, exist_ok=True)

        for subfolder in [checkpoint_folder, generation_folder]:
            os.makedirs(subfolder, exist_ok=True)

    return xp_folder, checkpoint_folder, generation_folder


def get_cosine_with_warmup_scheduler(optimizer, num_warmup_steps, num_training_steps, eta_min_ratio=0.1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        progress = min(1.0, max(0.0, progress))
        
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
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


def evaluate_model_loss(model, dataloader, criterion, local_rank):
    model.eval()
    local_loss_sum = 0.0
    local_batches = 0

    is_master = (local_rank == 0)
    iterator = tqdm(dataloader, total=len(dataloader), desc="Eval") if is_master else dataloader

    with torch.inference_mode():
        for batch in iterator:
            _, input_tokens, target_tokens, _, txt_tokens, txt_tokens_mask = batch

            input_tokens = input_tokens.to(local_rank).long()
            txt_tokens = txt_tokens.to(local_rank)
            txt_tokens_mask = txt_tokens_mask.to(local_rank)
            target_tokens = target_tokens.to(local_rank)

            logits = model(src=txt_tokens, tgt=input_tokens, src_mask=txt_tokens_mask)
            
            flat_logits = logits.reshape(-1, model.module.vocab_size)
            flat_targets = target_tokens.reshape(-1)

            loss = criterion(flat_logits, flat_targets)
            local_loss_sum += loss.item()
            local_batches += 1


    # Reduce losses across all visible GPUs
    t_loss = torch.tensor(local_loss_sum, device=local_rank)
    t_count = torch.tensor(local_batches, device=local_rank)
    
    dist.all_reduce(t_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(t_count, op=dist.ReduceOp.SUM)
    
    global_loss = (t_loss / t_count).item() if t_count > 0 else float("nan")
    model.train()
    
    return global_loss


def run_generation_test(test_dataloader, model, audio_tokenizer, generation_folder, local_rank):
    model.eval()
    iter_loader = iter(test_dataloader)
    batch = next(iter_loader)
    file_name_prefix, _, _, meta, txt_tokens, txt_tokens_mask = batch

    for i in range(len(file_name_prefix)):
        conditioner = txt_tokens[i].unsqueeze(0).to(local_rank)
        conditioner_mask = txt_tokens_mask[i].unsqueeze(0).to(local_rank)

        with torch.inference_mode():
            logger.info(f"Generating audio tokens for {file_name_prefix[i]}...")
            
            audio_tokens = model.module.generate(
                max_new_tokens=model.module.max_seq_len,
                src=conditioner,
                src_mask=conditioner_mask,
                temperature=2.0,
                top_k=250,
                top_p=0.9
            )

            print(f"Generate audio codes shape: {audio_tokens.shape}\n")

            total_generated_samples = audio_tokens.shape[-1] * audio_tokenizer.model.hop_length
            meta[i]['original_length'] = min(meta[i]['original_length'], total_generated_samples)
            decoded_audio = audio_tokenizer.decode(audio_tokens.cpu(), meta[i])

            logger.info(f"Decoded audio shape: {decoded_audio.shape}")
            path = os.path.join(generation_folder, file_name_prefix[i]+'.wav')
            decoded_audio.write(path)

    model.train()

def cleanup_and_exit(signum, frame):
    """Function triggered when program is terminated (including by Slurm)."""
    logger.info(f"Terminating this job... (signum, frame): ({signum}, {frame})")

    dist.destroy_process_group()
    sys.exit(0)


def _train_audio_worker(
        local_rank: int,
        world_size: int,
        dataset_class,
        dataset_root: str, 
        epochs: int, 
        warmup_steps: int,
        decoder_size: MusicGenSize,
        generate_every: int,
        audio_secs: int,
        xp_dir: str
    ):
    # Initialize DDP process group for this rank
    if not dist.is_initialized():
        os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
        os.environ.setdefault('MASTER_PORT', '29500')
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=local_rank,
            world_size=world_size
        )

    signal.signal(signal.SIGTERM, cleanup_and_exit)

    torch.cuda.set_device(local_rank)
    is_master = (local_rank == 0)

    # Instantiate directories (only rank 0 creates physical directories)
    if is_master:
        xp_folder, checkpoint_folder, generation_folder = instantiate_xp(xp_dir)
    else:
        xp_folder = os.path.join(xp_dir, XP_NAME)
        checkpoint_folder = os.path.join(xp_folder, "checkpoints")
        generation_folder = os.path.join(xp_folder, "generation_tests")

    dist.barrier() # Sync so all GPUs wait for directories

    # Load audio tokenizer
    dac = DACCompressionModel.get_pretrained("44khz", device='cpu').eval()
    dac.set_num_codebooks(4)

    # Instantiate Transformer and wrap in DDP
    model = MusicGenTransformer(
        vocab_size=dac.vocab_size,
        frame_rate=int(dac.frame_rate),
        audio_duration=audio_secs,
        pad_token_id=dac.pad_token_id,
        eos_token_id=dac.eos_token_id,
        bos_token_id=dac.bos_token_id,
        model_size=decoder_size
    ).to(local_rank)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # Distributed datasets & samplers
    train_dataset = dataset_class(
        dataset_root=dataset_root, 
        split="train",
        num_codebooks=model.module.num_codebooks,
        audio_secs=audio_secs
    )
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=local_rank, 
        shuffle=True
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=12,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate,
    )

    test_dataset = dataset_class(
        dataset_root=dataset_root, 
        split="test",
        num_codebooks=model.module.num_codebooks,
        audio_secs=audio_secs
    )
    test_sampler = DistributedSampler(
        test_dataset, 
        num_replicas=world_size, 
        rank=local_rank, 
        shuffle=False
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=12,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate,
    )

    # Optimization
    criterium = torch.nn.CrossEntropyLoss(ignore_index=model.module.pad_token_id)
    optim = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.95), weight_decay=0.1)
    scheduler = get_cosine_with_warmup_scheduler(optim, warmup_steps, epochs * len(train_dataloader))

    best_test_loss = float("inf")
    train_losses = []
    test_losses = []
    current_epoch = 0

    checkpoint_last_path = os.path.join(checkpoint_folder, f'checkpoint_last.pth')
    checkpoint_best_path = os.path.join(checkpoint_folder, f'checkpoint_best.pth')

    if os.path.isfile(checkpoint_last_path):
        ckpt = torch.load(checkpoint_last_path, map_location=f'cuda:{local_rank}', weights_only=False)

        model.module.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optim"])
        scheduler.load_state_dict(ckpt["scheduler"])

        best_test_loss = ckpt["best_test_loss"]
        train_losses = ckpt["loss_history"]["train"]
        test_losses = ckpt["loss_history"]["test"]
        current_epoch = ckpt["epoch"] + 1

        if is_master:
            logger.info(f"Loaded checkpoint from {checkpoint_last_path} | Current Epoch: {current_epoch} | Best Test Loss {best_test_loss:.4f}")

    dist.barrier()

    for epoch in range(current_epoch, epochs):
        if not model.training: model.train()

        train_sampler.set_epoch(epoch)
        epoch_losses = []

        iterator = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {epoch}") if is_master else train_dataloader

        for update, batch in enumerate(iterator):
            _, input_tokens, target_tokens, _, txt_tokens, txt_tokens_mask = batch

            input_tokens = input_tokens.to(local_rank).long()
            target_tokens = target_tokens.to(local_rank).long()

            # Create a tensor to hold the decision
            drop_decision = torch.zeros(1, device=local_rank)

            if is_master:
                if random.random() < 0.3:
                    drop_decision[0] = 1.0

            # Broadcast the master's decision to all GPUs
            dist.broadcast(drop_decision, src=0)

            if drop_decision.item() == 1.0:
                drop_conditioning = True
                txt_tokens = None
                txt_tokens_mask = None
            else:
                drop_conditioning = False
                txt_tokens = txt_tokens.to(local_rank)
                txt_tokens_mask = txt_tokens_mask.to(local_rank)

            optim.zero_grad()
            logits = model(src=txt_tokens, src_mask=txt_tokens_mask, tgt=input_tokens, drop_conditioning=drop_conditioning)

            flat_logits = logits.reshape(-1, model.module.vocab_size)
            flat_targets = target_tokens.reshape(-1)

            loss = criterium(flat_logits, flat_targets)
            loss_item = loss.item()

            if is_master:
                logger.debug(f"Epoch {epoch} -> Update {update} | Loss: {loss_item:.4f}")
            epoch_losses.append(loss_item)

            loss.backward()
            optim.step()
            scheduler.step()

        # Compute average loss across all GPUs
        t_epoch_loss = torch.tensor(sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0, device=local_rank)
        dist.all_reduce(t_epoch_loss, op=dist.ReduceOp.AVG)
        epoch_loss = t_epoch_loss.item()
        train_losses.append(epoch_loss)

        test_loss = evaluate_model_loss(model, test_dataloader, criterium, local_rank)
        test_losses.append(test_loss)

        if is_master:
            save_loss_history(xp_folder, train_losses, test_losses)
            logger.info(f"Epoch {epoch} Train Loss: {epoch_loss:.4f} | Test Loss: {test_loss:.4f}")

            ckpt = {
                "model": model.module.state_dict(),
                "optim": optim.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_test_loss": best_test_loss,
                "loss_history": {
                    "train": train_losses,
                    "test": test_losses
                }
            }
            torch.save(ckpt, checkpoint_last_path)

            if test_loss <= best_test_loss:
                logger.info(f"Saved BEST checkpoint to {checkpoint_best_path} | Current Epoch: {epoch} | Best Test Loss {test_loss:.4f}")
                best_test_loss = test_loss
                ckpt["best_test_loss"] = best_test_loss
                torch.save(ckpt, checkpoint_best_path)

            if generate_every > 0 and epoch % generate_every == 0:
                gen_epoch_folder = os.path.join(generation_folder, f"epoch_{epoch}")
                os.makedirs(gen_epoch_folder, exist_ok=True)
                run_generation_test(test_dataloader, model, dac, gen_epoch_folder, local_rank)

        dist.barrier()

    dist.destroy_process_group()


def train_audio(
        dataset_class,
        dataset_root: str, 
        epochs: int, 
        warmup_steps: int,
        decoder_size: MusicGenSize,
        generate_every: int = -1,
        audio_secs: int = 60,
        xp_dir = './'
    ):
    visible_gpus = torch.cuda.device_count()
    if visible_gpus == 0:
        raise RuntimeError("No visible CUDA GPUs found")

    # Check if process group environment variables exist (e.g., launched via torchrun)
    if "WORLD_SIZE" in os.environ and "RANK" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ["WORLD_SIZE"])
        _train_audio_worker(
            local_rank,
            world_size,
            dataset_class,
            dataset_root,
            epochs,
            warmup_steps,
            decoder_size,
            generate_every,
            audio_secs,
            xp_dir
        )
    else:
        # Automatically spawn worker processes for every visible GPU
        print(f"Detected {visible_gpus} visible GPU(s). Spawning DDP processes...")
        mp.spawn(
            _train_audio_worker,
            args=(
                visible_gpus,
                dataset_class,
                dataset_root,
                epochs,
                warmup_steps,
                decoder_size,
                generate_every,
                audio_secs,
                xp_dir
            ),
            nprocs=visible_gpus,
            join=True
        )