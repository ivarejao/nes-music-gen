from miditok import MMM
from pathlib import Path
import json

def test_detokenizer():
    # 1. Define os caminhos base
    caminho_base = Path("any2music/symbolic")
    arquivo_tokenizador = caminho_base / "nes_tokenizers" / "nes_mmm_tokenizer.json"
    
    # Cria uma pasta para salvar as músicas geradas/reconstruídas
    pasta_saida = caminho_base / "generated_midi"
    pasta_saida.mkdir(parents=True, exist_ok=True)
    
    # 2. Carrega o tokenizador com as suas regras de 8-bits
    print("Carregando o Destokenizador...")
    tokenizer = MMM(params=arquivo_tokenizador)
    
    # 3. Pega um arquivo JSON de teste que você gerou no passo anterior
    pasta_tokens_teste = caminho_base / "nesmdb_tokens" / "test"
    arquivos_json = list(pasta_tokens_teste.glob("*.json"))
    
    if not arquivos_json:
        print("Nenhum arquivo JSON encontrado na pasta de teste.")
        return
        
    arquivo_alvo = arquivos_json[0] # Pega o primeiro arquivo da lista
    print(f"\nLendo tokens (números) do arquivo: {arquivo_alvo.name}")
    
    with open(arquivo_alvo, 'r') as f:
        ids_numericos = json.load(f)
        
    print(f"Total de tokens lidos: {len(ids_numericos)}")

    try:
        # 4. A MÁGICA: Converte os números de volta para música!
        # O MidiTok entende que se você passar uma lista de inteiros, ele deve fazer o "decode"
        midi_reconstruido = tokenizer(ids_numericos)
        
        # 5. Salva o resultado
        caminho_salvar = pasta_saida / f"reconstruido_{arquivo_alvo.stem}.mid"
        
        # A biblioteca symusic (que roda por baixo do MidiTok v3) usa o método dump_midi
        midi_reconstruido.dump_midi(caminho_salvar)
        
        print(f"\nSucesso! Música reconstruída e salva em: {caminho_salvar}")
        print("Abra esse arquivo no seu player de MIDI ou na sua DAW para ouvir!")
        
    except Exception as e:
        print(f"Erro na destokenização: {e}")

if __name__ == "__main__":
    test_detokenizer()