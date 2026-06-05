from miditok import MMM
from pathlib import Path
import json

def process_dataset():
    caminho_base = Path("any2music/symbolic")
    pasta_midi = caminho_base / "nesmdb_midi"
    pasta_tokens = caminho_base / "nesmdb_tokens"
    
    arquivo_tokenizador = caminho_base / "nes_tokenizers" / "nes_mmm_tokenizer.json"

    print(f"Carregando tokenizador de: {arquivo_tokenizador}")
    tokenizer = MMM(params=arquivo_tokenizador)

    splits = ["train", "valid", "test"]

    for split in splits:
        pasta_midi_split = pasta_midi / split
        pasta_tokens_split = pasta_tokens / split
        
        pasta_tokens_split.mkdir(parents=True, exist_ok=True)
        
        arquivos_midi = list(pasta_midi_split.glob("*.mid"))
        
        if not arquivos_midi:
            print(f"Aviso: Nenhum arquivo MIDI encontrado em {pasta_midi_split}")
            continue
            
        print(f"\nProcessando {len(arquivos_midi)} arquivos da pasta '{split}'...")
        
        for arquivo in arquivos_midi:
            try:
                tokens = tokenizer(arquivo)
                
                if isinstance(tokens, list):
                    ids_numericos = tokens[0].ids 
                else:
                    ids_numericos = tokens.ids 
                
                caminho_salvar = pasta_tokens_split / f"{arquivo.stem}.json"
                
                with open(caminho_salvar, 'w') as f:
                    json.dump(ids_numericos, f)
                    
            except Exception as e:
                print(f"Erro ao processar {arquivo.name}: {e}")

    print("\nExtração concluída com sucesso!")

if __name__ == "__main__":
    process_dataset()