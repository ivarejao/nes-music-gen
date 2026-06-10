import pickle
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

def bpe_training():
    root = Path(__file__).resolve().parent.parent.parent
    pasta_treino_pkl = root / "data" / "nesmdb_nlm" / "train"
    path_bpe = root / "models" / "tokenizers" / "lmf_bpe_373_tokenizer.json"
    pasta_temp_txt = root / "data" / "temp_bpe_corpus"
    
    arquivos_pkl = list(pasta_treino_pkl.glob("*.pkl"))
    if not arquivos_pkl:
        print("")
        return
    
    pasta_temp_txt.mkdir(parents=True, exist_ok=True)
    corpus_path = pasta_temp_txt / "lmf_corpus_raw.txt"
    
    with open(corpus_path, 'w', encoding='utf-8') as f_out:
        for arquivo in arquivos_pkl:
            with open(arquivo, 'rb') as f_in:
                dados_nlm = pickle.load(f_in)
                linha = " ".join([f"{cmd[0]}_{cmd[1]}" for cmd in dados_nlm])
                f_out.write(linha + "\n")
    
    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
    
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    trainer = trainers.BpeTrainer(
        vocab_size=373,
        special_tokens=["<PAD>", "<BOS>", "<EOS>", "<UNK>"],
        show_progress=True
    )
    
    tokenizer.train(files=[str(corpus_path)], trainer=trainer)
    
    path_bpe.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(path_bpe))

    print(f"{trainer.vocab_size} tokens")

if __name__ == "__main__":
    bpe_training()