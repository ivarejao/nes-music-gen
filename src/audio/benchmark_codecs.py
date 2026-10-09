import os
import argparse
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
from tqdm import tqdm
from collections import defaultdict
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
import auraloss
from visqol import VisqolApi
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

def parse_args():
    """Imita a estrutura de argumentos de linha de comando do repositório DAC."""
    parser = argparse.ArgumentParser(description="Evaluate Audio Codecs (DAC-style)")
    parser.add_argument("--original-dir", type=str, required=True, help="Path to original audio files")
    parser.add_argument("--reconstructed-dir", type=str, required=True, help="Path to reconstructed audio files")
    parser.add_argument("--target-sr", type=int, default=44100, help="Target sample rate for evaluation")
    return parser.parse_args()

def setup_metrics(target_sr):
    """Instancia as métricas espelhando os parâmetros exatos do artigo do DAC."""
    metrics_dict = {
        "si_sdr": ScaleInvariantSignalDistortionRatio().cpu(),
        "db": T.AmplitudeToDB()
    }
    
    # STFT Distance conforme Seção 4.4 do paper: "We use window lengths [2048, 512]"
    # Como padrão do artigo, o hop_length é 1/4 do win_length.
    metrics_dict["stft"] = auraloss.freq.MultiResolutionSTFTLoss(
        fft_sizes=[2048, 512],
        hop_sizes=[512, 128], 
        win_lengths=[2048, 512]
    ).cpu()

    # Mel Distance conforme Seção 3.5 do paper (Multi-Scale)
    # window lengths: [32, 64, 128, 256, 512, 1024, 2048]
    # mels: [5, 10, 20, 40, 80, 160, 320]
    mel_transforms = []
    win_lengths = [32, 64, 128, 256, 512, 1024, 2048]
    n_mels_list = [5, 10, 20, 40, 80, 160, 320]
    
    for win, mels in zip(win_lengths, n_mels_list):
        mel_transforms.append(
            T.MelSpectrogram(
                sample_rate=target_sr,
                n_fft=win,
                win_length=win,
                hop_length=win // 4, # O paper cita explicitly: hop length set to window_length / 4
                n_mels=mels
            )
        )
    metrics_dict["mel_transforms"] = mel_transforms
    
    # Inicializa a API do ViSQOL para avaliação perceptiva (Seção 4.4)
    visqol_api = VisqolApi()
    visqol_api.create(mode="audio")
    metrics_dict["visqol_api"] = visqol_api
    
    return metrics_dict

def load_and_align(orig_path, recon_path, target_sr):
    """Equivalente ao comportamento de carregamento da classe AudioSignal da audiotools."""
    wav_orig, sr_orig = torchaudio.load(orig_path)
    wav_recon, sr_recon = torchaudio.load(recon_path)

    # Downmix para Mono
    if wav_orig.shape[0] > 1: wav_orig = torch.mean(wav_orig, dim=0, keepdim=True)
    if wav_recon.shape[0] > 1: wav_recon = torch.mean(wav_recon, dim=0, keepdim=True)
        
    # Resample
    if sr_orig != target_sr: wav_orig = F.resample(wav_orig, sr_orig, target_sr)
    if sr_recon != target_sr: wav_recon = F.resample(wav_recon, sr_recon, target_sr)
        
    # Alinhamento temporal 
    min_frames = min(wav_orig.shape[-1], wav_recon.shape[-1])
    return wav_orig[..., :min_frames], wav_recon[..., :min_frames]

def compute_visqol(wav_orig, wav_recon, current_sr, visqol_api):
    """Isola a lógica do ViSQOL que a audiotools faria nativamente em C++."""
    ref_48k = F.resample(wav_orig, current_sr, 48000)
    deg_48k = F.resample(wav_recon, current_sr, 48000)

    temp_ref, temp_deg = "temp_ref.wav", "temp_deg.wav"
    torchaudio.save(temp_ref, ref_48k, 48000)
    torchaudio.save(temp_deg, deg_48k, 48000)

    try:
        moslqo = visqol_api.measure(temp_ref, temp_deg).moslqo
    finally:
        if os.path.exists(temp_ref): os.remove(temp_ref)
        if os.path.exists(temp_deg): os.remove(temp_deg)

    return moslqo

def evaluate(args):
    """O loop de avaliação principal, espelhando o evaluate.py do DAC."""
    print(f"Evaluating {args.reconstructed_dir} at {args.target_sr}Hz")
    
    metrics = setup_metrics(args.target_sr)
    metrics_state = defaultdict(list) # Armazena todos os valores como o código original
    
    orig_files = {f.split('.')[0]: f for f in os.listdir(args.original_dir) if f.endswith('.wav') or f.endswith('.mp3')}
    recon_files = [f for f in os.listdir(args.reconstructed_dir) if f.endswith('.wav')]
    
    for recon_file in tqdm(recon_files, desc="Processing Audio"):
        prefix = recon_file.split('.')[0]
        if prefix not in orig_files: continue
            
        orig_path = os.path.join(args.original_dir, orig_files[prefix])
        recon_path = os.path.join(args.reconstructed_dir, recon_file)
        
        try:
            wav_orig, wav_recon = load_and_align(orig_path, recon_path, args.target_sr)
            
            # Batch dimension para o PyTorch
            w_o_batch = wav_orig.unsqueeze(0)
            w_r_batch = wav_recon.unsqueeze(0)

            # --- Cálculos das Métricas ---
            
            # 1. SI-SDR
            val_si_sdr = metrics["si_sdr"](w_r_batch.squeeze(1), w_o_batch.squeeze(1)).item()
            metrics_state["si_sdr"].append(val_si_sdr)
            
            # 2. STFT Distance
            val_stft = metrics["stft"](w_r_batch, w_o_batch).item()
            metrics_state["stft_distance"].append(val_stft)

            # ... (SI-SDR e STFT continuam iguais) ...

            # 3. Mel Distance (Multi-Scale conforme Seções 3.5 e 4.4)
            # Soma do erro absoluto (L1) entre os log-mels de todas as 7 escalas
            val_mel_total = 0.0
            for mel_tfm in metrics["mel_transforms"]:
                mel_o = metrics["db"](mel_tfm(wav_orig))
                mel_r = metrics["db"](mel_tfm(wav_recon))
                val_mel_total += torch.abs(mel_o - mel_r).mean().item()

            val_mel_mean = val_mel_total / len(metrics["mel_transforms"])
            metrics_state["mel_distance"].append(val_mel_mean)

            # 4. ViSQOL
            val_visqol = compute_visqol(wav_orig, wav_recon, args.target_sr, metrics["visqol_api"])
            metrics_state["visqol"].append(val_visqol)

        except Exception as e:
            print(f"Skipping {prefix} due to error: {e}")

    # Agregação e Relatório (Estilo DAC)
    print("\n" + "="*50)
    print("FINAL EVALUATION METRICS")
    print("="*50)
    for metric_name, values in metrics_state.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{metric_name.ljust(15)} : {mean_val:.4f} ± {std_val:.4f}")
    print("="*50)

if __name__ == '__main__':
    torch.set_num_threads(4) 
    args = parse_args()
    evaluate(args)