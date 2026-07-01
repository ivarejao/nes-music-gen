import os
import gc
import glob
import json
import argparse
import warnings

import torch
import torchaudio

from src.midi.MIDI import MIDIHandler
from src.nlm.NLM import NLMHandler

from any2music.metrics.fad import FADMetric
from any2music.metrics.kld import KLDMetric
from any2music.metrics.clap_score import ClapScoreMetric

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Maps each metric to its factory and the key returned by compute().
METRIC_KEYS = {
    "kld": "kld",
    "fad": "fad_score",
    "clap": "clap_score",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Music Generation Benchmark (NESMDB)")
    
    parser.add_argument("--gt_dir", type=str, required=True, 
                        help="Directory containing the ground-truth .wav files.")
    
    parser.add_argument("--pred_wavs", type=str, default=None, 
                        help="Directory containing the generated .wav files.")
    parser.add_argument("--pred_midis", type=str, default=None, 
                        help="Directory containing the generated .mid files (will use MIDIHandler).")
    parser.add_argument("--pred_nlms", type=str, default=None, 
                        help="Directory containing the generated NLM files (will use NLMHandler).")
    
    parser.add_argument("--midi_tokenizer", type=str, 
                        default="models/tokenizers/symbolic/midi_mmm_tokenizer.json",
                        help="Path to the MIDI tokenizer JSON file.")
    parser.add_argument("--nlm_tokenizer", type=str, 
                        default="models/tokenizers/symbolic/nlm_tokenizer.json",
                        help="Path to the NLM tokenizer JSON file.")
    
    parser.add_argument("--captions", type=str, default=None, 
                        help="JSON {filename: text} or directory with .txt files used for the CLAP score.")

    
    return parser.parse_args()


def free_gpu(*objs):
    """Drop metric objects and clear the CUDA cache so the GPU is free."""
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def load_wav(path):
    """Load a wav as [1, Channels, Time] (batch dim added), plus its sample rate."""
    wav, sr = torchaudio.load(path)          # [Channels, Time]
    return wav.unsqueeze(0), sr              # [1, Channels, Time]

def collect_pairs(gt_dir, pred_dir):
    """Match ground-truth and predicted wavs by file name."""
    gt_files = {os.path.basename(p): p for p in glob.glob(os.path.join(gt_dir, "*.wav"))}
    pred_files = {os.path.basename(p): p for p in glob.glob(os.path.join(pred_dir, "*.wav"))}

    common = sorted(set(gt_files) & set(pred_files))
    if not common:
        raise RuntimeError("No matching .wav files found between the two directories.")

    missing = sorted(set(gt_files) ^ set(pred_files))
    if missing:
        print(f"[Warning] {len(missing)} file(s) not present in both dirs, skipped: {missing[:5]}...")

    return [(name, gt_files[name], pred_files[name]) for name in common]

def load_captions(captions_path):
    """
    Load captions. Supports either:
      - a directory of per-file .txt captions (e.g. captions/<name>.txt), or
      - a single JSON file {filename: text}.
    Returns a dict keyed by base name WITHOUT extension.
    """
    captions = {}
    if not captions_path:
        return captions

    if os.path.isdir(captions_path):
        for txt in glob.glob(os.path.join(captions_path, "*.txt")):
            key = os.path.splitext(os.path.basename(txt))[0]
            with open(txt, "r", encoding="utf-8") as f:
                captions[key] = f.read().strip()
    elif os.path.isfile(captions_path):
        with open(captions_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # Normalize JSON keys to base names without extension too.
        captions = {os.path.splitext(k)[0]: v for k, v in raw.items()}

    return captions

def mean_std(values):
    """Population-friendly mean/std (std=0 when only one sample)."""
    if not values:
        return float("nan"), float("nan")
    t = torch.tensor(values, dtype=torch.float64)
    std = t.std(unbiased=True).item() if t.numel() > 1 else 0.0
    return t.mean().item(), std

def feed_pair(metric, name, gt_path, pred_path, captions, needs_text):
    """Load and update a single metric with one file pair. Returns False if skipped."""
    preds, sr = load_wav(pred_path)
    targets, sr_t = load_wav(gt_path)

    if sr != sr_t:                                  # keep sample rates consistent
        preds = torchaudio.functional.resample(preds, sr, sr_t)
        sr = sr_t

    if needs_text:
        key = os.path.splitext(name)[0]          # '...PowerAttackBoss.wav' -> '...PowerAttackBoss'
        text = captions.get(key)
        if text is None:
            print(f"[warn] no caption for '{name}', skipping it for CLAP.")
            return False
        metric.update(preds, targets, current_sr=sr, texts=[text])
    else:
        metric.update(preds, targets, current_sr=sr)
    return True

def run_metric(metric_name, metric_factory, pairs, captions=None, needs_text=False):
    """
    Compute the overall score (all files together) AND a per-file score
    distribution so we can report mean ± std. The GPU is freed afterwards.
    """
    score_key = METRIC_KEYS[metric_name]

    # --- Overall score: feed every pair into one metric instance ---
    overall_metric = metric_factory()
    used = 0
    for name, gt_path, pred_path in pairs:
        if feed_pair(overall_metric, name, gt_path, pred_path, captions, needs_text):
            used += 1
    overall = overall_metric.compute()[score_key] if used else float("nan")
    free_gpu(overall_metric)

    # --- Per-file scores: fresh metric instance per pair for the std ---
    per_file = []
    for name, gt_path, pred_path in pairs:
        m = metric_factory()
        ok = feed_pair(m, name, gt_path, pred_path, captions, needs_text)
        if ok:
            try:
                per_file.append(m.compute()[score_key])
            except Exception as e:
                print(f"[warn] per-file {metric_name} failed for '{name}': {e}")
        free_gpu(m)

    mean, std = mean_std(per_file)
    return {"overall": overall, "mean": mean, "std": std, "n": len(per_file)}

def benchmark(gt_dir, pred_dir, captions_path=None):
    print("=== Music Generation Benchmark (KLD → FAD → CLAPScore) ===")
    pairs = collect_pairs(gt_dir, pred_dir)
    print(pairs)
    captions = load_captions(captions_path)
    print(f"Found {len(pairs)} matched file pair(s).\n")

    print("Computing KLD...")
    kld_res = run_metric("kld", KLDMetric, pairs)

    print("Computing FAD...")
    fad_res = run_metric("fad", FADMetric, pairs)

    print("Computing CLAPScore...")
    clap_res = run_metric("clap", ClapScoreMetric, pairs, captions=captions, needs_text=True)

    print("\n==== RESULTS (overall | per-file mean ± std) ====")
    for label, res in [("KLD ", kld_res), ("FAD ", fad_res), ("CLAP", clap_res)]:
        print(f"{label}: {res['overall']:.6f} | "
              f"{res['mean']:.6f} ± {res['std']:.6f}  (n={res['n']})")

    return {"kld": kld_res, "fad": fad_res, "clap": clap_res}

def synthesize_files(handler, input_dir, output_dir, ext):
    """
    Scan the input directory for a specific extension, synthesize 
    each file via the Docker handler, and save the resulting WAVs.
    """
    input_path = os.path.abspath(input_dir)
    output_path = os.path.abspath(output_dir)
    os.makedirs(output_path, exist_ok=True)
    
    files = glob.glob(os.path.join(input_path, f"*.{ext}"))
    if not files:
        print(f"[Warning] No .{ext} files found in {input_path}")
        return output_path

    print(f"Synthesizing {len(files)} .{ext.upper()} files via Docker container...")
    for file_path in files:
        handler.synthesize(file_path, output_path)
        
    return output_path

def run_symbolic_pipeline(handler_class, tokenizer_path, pred_dir, gt_dir, captions, label, ext):
    """
    Initializes the specific handler, validates paths against the Docker mount point,
    triggers the batch synthesis, and executes the audio metrics benchmark.
    """
    print(f"Processing {label.upper()} pipeline...")
    handler = handler_class(tokenizer_path=tokenizer_path)
    
    if not handler.api_running or not handler.data_mount_point:
        raise RuntimeError(f"NESMDB API is offline or DATA_MOUNT_POINT is not accessible for {label.upper()}.")
        
    input_abs = os.path.abspath(pred_dir)
    if not input_abs.startswith(os.path.abspath(handler.data_mount_point)):
        raise ValueError(f"The input directory {input_abs} must be inside the DATA_MOUNT_POINT.")
        
    out_dir = os.path.join(handler.data_mount_point, f"eval_benchmark_{label}")
    
    wav_dir = synthesize_files(handler, pred_dir, out_dir, ext)
    benchmark(gt_dir, wav_dir, captions)

def main():
    args = parse_args()
    
    # Pipeline 1: Direct WAV Evaluation
    if args.pred_wavs:
        print("Evaluating direct WAV predictions...")
        benchmark(args.gt_dir, args.pred_wavs, args.captions)
        
    # Pipeline 2: Symbolic MIDI Evaluation
    if args.pred_midis:
        run_symbolic_pipeline(
            handler_class=MIDIHandler,
            tokenizer_path=args.midi_tokenizer,
            pred_dir=args.pred_midis,
            gt_dir=args.gt_dir,
            captions=args.captions,
            label="midi",
            ext="mid"
        )
        
    # Pipeline 3: Text NLM Evaluation
    if args.pred_nlms:
        run_symbolic_pipeline(
            handler_class=NLMHandler,
            tokenizer_path=args.nlm_tokenizer,
            pred_dir=args.pred_nlms,
            gt_dir=args.gt_dir,
            captions=args.captions,
            label="nlm",
            ext="nlm"
        )


if __name__ == "__main__":
    main()