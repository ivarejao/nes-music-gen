from miditok import MMM, TokenizerConfig
from pathlib import Path

def create_nes_mmm_tokenizer():
    config = TokenizerConfig(
        pitch_range=(33, 107), 
        beat_res={(0, 1): 8, (1, 2): 4, (2, 4): 2, (4, 8): 1}, 
        num_velocities=16, 
        use_chords=False,    
        use_programs=True,
        use_tempos=True
    )
    
    config.additional_params["base_tokenizer"] = "REMI"


    tokenizer = MMM(config)
    return tokenizer

if __name__ == "__main__":
    nes_tokenizer = create_nes_mmm_tokenizer()
    print("Tokenizador MMM (Base REMI) criado com sucesso!")
    print(f"Tamanho do vocabulário fixo: {len(nes_tokenizer)} tokens.")

    caminho_salvamento = Path("any2music/symbolic/nes_tokenizers/nes_mmm_tokenizer.json")
    caminho_salvamento.parent.mkdir(parents=True, exist_ok=True)
    
    nes_tokenizer.save_params(caminho_salvamento)
    print(f"Tokenizador salvo em: {caminho_salvamento}")