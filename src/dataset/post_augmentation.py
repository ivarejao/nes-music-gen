import os 
import pandas as pd
from argparse import ArgumentParser, Namespace
from pathlib import Path

from src.midi.MIDI import MIDIHandler
from src.nlm.NLM import NLMHandler

def parse_args() -> Namespace:
    parser = ArgumentParser(description='Post processing after main augmentation script is done')
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--midi-tokenizer', help='Define midi-tokenizer')
    parser.add_argument('--nlm-tokenizer', help='Define nlm-tokenizer')

    args = parser.parse_args()
    return args



# 1. Walk through the dataset, only walk on the new subsamples
# 2. For each new sample:
# 2.a. Tokenize the midi and NLM with the name {track_file_name}_{format}_tokens.json
# 2.b. Tag the new samples
# 3. Join the new metadata


# MIDI
# try:
#     tokens = midi_tokenizer.tokenize(arquivo)
    
#     if isinstance(tokens, list):
#         ids_numericos = tokens[0].ids 
#     else:
#         ids_numericos = tokens.ids 
    
#     # caminho_salvar = pasta_tokens_split / f"{arquivo.stem}.npy"
    
#     # with open(caminho_salvar, 'wb') as f:
#     #     np.save(f, ids_numericos)
#     caminho_salvar = pasta_tokens_split / f"{arquivo.stem}.json"
    
#     with open(caminho_salvar, 'w') as f:
#         json.dump(ids_numericos, f)
        

# except Exception as e:
#     print(f"Erro ao processar {arquivo.name}: {e}")


# NLM
# try:
#     tokens = nlm_tokenizer.tokenize(pkl_file)
#     ids = tokens.ids

#     clean_name = pkl_file.name.replace(".nlm.pkl", "")
#     save_path  = output_folder / f"{clean_name}.json"

#     with open(save_path, "w") as f:
#         json.dump(ids, f)

# except Exception as e:
#     print(f"Error processing {pkl_file.name}: {e}")



def tokenize(file_path, handler, format : str):
    try:
        tokens = handler.tokenize(file_path)
        ids = tokens.ids

        clean_name = pkl_file.name.replace(".nlm.pkl", "")
        save_path  = output_folder / f"{clean_name}.json"

        with open(save_path, "w") as f:
            json.dump(ids, f)

    except Exception as e:
        print(f"Error processing {pkl_file.name}: {e}")




def post_processing_dataset(dataset_path : Path, nlm_tokenizer_path : Path, midi_tokenizer_path):

    dataset_path = Path(dataset_path)

    nlm_handler = NLMHandler(tokenizer_path=nlm_tokenizer_path)
    midi_handler = MIDIHandler(tokenizer_path=midi_tokenizer_path)


    new_samples_pattern = \
        r".*(_b_[0-9]+|_#_[0-9]+|_original)" \
        r"(_slow_[0-9]+|_fast_[0-9]+|_original)'"

    for track_path in dataset_path.glob(new_samples_pattern):

        if track_path.is_dir():
            game_id = track_path.parent.name
            track_name = track_path.name

            for file_path in track_path.iterdir():
                if file_path.is_file():
                    ext = file_path.name.rsplit(".")[-1]
                    if ext == ".mid":
                        midi_handler.tokenize(str(file_path))
                        pass
                    elif ext == ".nlm.pkl":
                        pass
                    elif ext == ".wav":
                        # TODO: revisit and check it
                        # this is just a fix from the augmentation main script
                        new_name = file_path.name.replace(".mid", "")
                        file_path.replace(new_name)








def main():
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    nlm_tokenizer_path = Path(args.nlm_tokenizer)
    midi_tokenizer_path = Path(args.midi_tokenizer)

    post_processing_dataset(dataset_path, nlm_tokenizer_path, midi_tokenizer_path)

if __name__ == "__main__":
    main()