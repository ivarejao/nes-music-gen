from pathlib import Path
import numpy as np
from argparse import Namespace, ArgumentParser
from tqdm import tqdm
from tokenizer import MIDITokenizer


import json

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--tokenizer-path')
    parser.add_argument('--output-dir', default='out/')
    parser.add_argument('--tokenize', '-t', action='store_true', help='Apply tokenization')
    parser.add_argument('--detokenize', '-dt', action='store_true', help='Apply detokenization')

    args = parser.parse_args()
    return args

def tokenize_dataset(data_dir : Path, midi_tokenizer : MIDITokenizer, output_dir : Path):
    print("Starting tokenization")
    # ensure output_dir exists
    output_dir = output_dir / "midi_tokens"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    splits = ["train", "valid", "test"]

    for split in splits:
        pasta_midi_split = data_dir / split
        pasta_tokens_split = output_dir / split
        
        pasta_tokens_split.mkdir(parents=True, exist_ok=True)
        
        arquivos_midi = list(pasta_midi_split.glob("*.mid"))
        
        if not arquivos_midi:
            print(f"Aviso: Nenhum arquivo MIDI encontrado em {pasta_midi_split}")
            continue
            
        print(f"\nProcessando {len(arquivos_midi)} arquivos da pasta '{split}'...")
        
        for arquivo in tqdm(arquivos_midi):
            try:
                tokens = midi_tokenizer.tokenize(arquivo)
                
                if isinstance(tokens, list):
                    ids_numericos = tokens[0].ids 
                else:
                    ids_numericos = tokens.ids 
                
                # caminho_salvar = pasta_tokens_split / f"{arquivo.stem}.npy"
                
                # with open(caminho_salvar, 'wb') as f:
                #     np.save(f, ids_numericos)
                caminho_salvar = pasta_tokens_split / f"{arquivo.stem}.json"
                
                with open(caminho_salvar, 'w') as f:
                    json.dump(ids_numericos, f)
                    

            except Exception as e:
                print(f"Erro ao processar {arquivo.name}: {e}")

    print("\nExtração concluída com sucesso!")


def detokenize_dataset(data_dir : Path, midi_tokenizer : MIDITokenizer, output_dir : Path):
    
    print("Starting *de*-tokenization")
    # ensure output_dir exists
    output_dir = output_dir / "midi_decoded"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    tokens_files = list(data_dir.glob("*.json"))
    
    if not tokens_files:
        print(f"Nenhum arquivo .npy encontrado na pasta de {data_dr}.")
        return
            
    for file_path in tokens_files:
        print(f"\nLendo tokens (números) do arquivo: {file_path.name}")

        try:
            # with open(file_path, 'rb') as f: 
            #     numeric_ids = np.load(f)
            with open(file_path, 'r') as f:
                numeric_ids = json.load(f)
            print("Tokens: {}".format(len(numeric_ids)))
            print("raw value: {}".format(numeric_ids))
            midi_detokenized = midi_tokenizer.detokenize(numeric_ids)
            output_fpath = output_dir / f"{file_path.stem}_midi_decoded.mid"
            midi_detokenized.dump_midi(output_fpath)
            
            print(f"Sucess!! Music MIDI detokenized at {output_fpath}")
            print("Open this file on a DAW to hear")
            
        except Exception as e:
            print(f"Error while destokenizing {e}")

def main():
    args = parse_args()
    midi_tokenizer = MIDITokenizer(Path(args.tokenizer_path), log_file=Path(args.output_dir) / "log.txt")
    import json
    vocab_path = Path(args.output_dir) / "vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(midi_tokenizer.model.vocab, f)
    print("Loaded model with vocab at {}".format(vocab_path))
    if args.tokenize:
        tokenize_dataset(Path(args.data_dir), midi_tokenizer, Path(args.output_dir))
    if args.detokenize:
        detokenize_dataset(Path(args.output_dir) / "midi_tokens" / "test", midi_tokenizer, Path(args.output_dir))

if __name__ == "__main__":
    main()