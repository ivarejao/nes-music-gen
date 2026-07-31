import os
import glob
import argparse

from src.midi.MIDI import MIDIHandler
from src.nlm.NLM import NLMHandler

def parse_args():
    parser = argparse.ArgumentParser(description="Batch Synthesizer: MIDI/NLM to WAV via Docker")
    
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Directory containing the .mid or .nlm files.")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save the resulting .wav files.")
    parser.add_argument("--mode", type=str, choices=["midi", "nlm"], required=True, 
                        help="The input modality (determines which handler to use).")
    parser.add_argument("--tokenizer_path", type=str, required=True, 
                        help="Path to the tokenizer JSON configuration file.")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    input_abs = os.path.abspath(args.input_dir)
    output_abs = os.path.abspath(args.output_dir)
    os.makedirs(output_abs, exist_ok=True)
    
    if args.mode == "midi":
        print(f"Initializing MIDI Handler with tokenizer: {args.tokenizer_path}")
        handler = MIDIHandler(tokenizer_path=args.tokenizer_path)
        ext = "mid"
    elif args.mode == "nlm":
        print(f"Initializing NLM Handler with tokenizer: {args.tokenizer_path}")
        handler = NLMHandler(tokenizer_path=args.tokenizer_path)
        ext = "pkl"
        
    if not handler.api_running or not handler.data_mount_point:
        raise RuntimeError(f"NESMDB API is offline or DATA_MOUNT_POINT is not accessible.")
        
    mount_point_abs = os.path.abspath(handler.data_mount_point)
    
    if not input_abs.startswith(mount_point_abs):
        raise ValueError(f"Input directory {input_abs} MUST be inside DATA_MOUNT_POINT ({mount_point_abs}).")
    if not output_abs.startswith(mount_point_abs):
        raise ValueError(f"Output directory {output_abs} MUST be inside DATA_MOUNT_POINT ({mount_point_abs}).")

    files = glob.glob(os.path.join(input_abs, f"*.{ext}"))
    
    if not files:
        print(f"[Warning] No .{ext} files found in {input_abs}")
        return
        
    print(f"Synthesizing {len(files)} {args.mode.upper()} files into WAVs via Docker...")
    
    for file_path in files:
            base_name = os.path.basename(file_path)
            nome_sem_extensao = os.path.splitext(base_name)[0]
            

            nome_limpo = nome_sem_extensao.replace(".nlm", "") #conferir se replace é necessario
            
            wav_esperado = os.path.join(output_abs, f"{nome_limpo}.wav")
            
            # A Mágica do Skip:
            if os.path.exists(wav_esperado):
                print(f"[Skip] {nome_limpo}.wav já existe. Pulando...")
                continue
                
            try:
                handler.synthesize(file_path, output_abs)
            except Exception as e:
                print(f"[Error] Failed to synthesize {base_name}: {e}")
            
    print(f"Batch synthesis complete. WAVs saved to {output_abs}")

if __name__ == "__main__":
    main()