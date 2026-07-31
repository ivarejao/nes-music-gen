import os
import json
import glob
import argparse

from src.midi.MIDI import MIDIHandler
from src.nlm.NLM import NLMHandler

def parse_args():
    parser = argparse.ArgumentParser(description="Batch Detokenizer: JSON to MIDI/NLM")
    
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="Directory containing the tokenized .json files.")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save the resulting detokenized files.")
    parser.add_argument("--mode", type=str, choices=["midi", "nlm"], required=True, 
                        help="The target modality (determines which handler to use).")
    parser.add_argument("--tokenizer_path", type=str, required=True, 
                        help="Path to the tokenizer JSON configuration file.")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == "midi":
        print(f"Initializing MIDI Handler with tokenizer: {args.tokenizer_path}")
        handler = MIDIHandler(tokenizer_path=args.tokenizer_path)
        target_ext = "mid"
    elif args.mode == "nlm":
        print(f"Initializing NLM Handler with tokenizer: {args.tokenizer_path}")
        handler = NLMHandler(tokenizer_path=args.tokenizer_path)
        target_ext = "nlm"
        
    json_files = glob.glob(os.path.join(args.input_dir, "*.json"))
    
    if not json_files:
        print(f"[Warning] No .json files found in {args.input_dir}")
        return
        
    print(f"Detokenizing {len(json_files)} {args.mode.upper()} files...")
    
    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
                numeric_ids = data[0]
            else:
                numeric_ids = data
            
            detokenized_obj = handler.detokenize(numeric_ids)
            
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            
            if args.mode == "midi":
                detokenized_obj = handler.detokenize(numeric_ids)
                output_file_path = os.path.join(args.output_dir, f"{base_name}.mid")
                detokenized_obj.dump_midi(output_file_path)
                
            elif args.mode == "nlm":
                # 1. Detokeniza para string
                nlm_script = handler.detokenize(numeric_ids)
                # 2. Converte a string de volta para a estrutura original de tuplas
                nlm_tuples = handler.parse_nlm(nlm_script)
                # 3. Define a extensão correta exigida pela API do Docker
                output_file_path = os.path.join(args.output_dir, f"{base_name}.nlm.pkl")
                # 4. Usa o próprio Handler para persistir o Pickle corretamente
                handler.save_nlm(nlm_tuples, output_file_path)
                    
        except Exception as e:
            print(f"[Error] Failed to detokenize {os.path.basename(file_path)}: {e}")
            
    print(f"Batch detokenization complete. Output saved to {args.output_dir}")

if __name__ == "__main__":
    main()