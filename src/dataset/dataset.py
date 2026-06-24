import os
import logging
import random

import pandas as pd

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class NESDataset(Dataset):
    def __init__(self, dataset_root:str, split:str, num_codebooks, audio_secs:int) -> None:
        super().__init__()

        self.dataset_root = dataset_root
        self.csv_path = os.path.join(self.dataset_root, "nes_metadata.csv")
        self.num_codebooks = num_codebooks
        self.audio_secs = audio_secs

        self.df = pd.read_csv(self.csv_path, dtype={"id":"object"})

        if split == "train":
            self.df:pd.DataFrame = self.df[self.df.split.isin(["train", "valid"])]
        elif split == "test":
            self.df:pd.DataFrame = self.df[self.df["split"] == "test"]
        else:
            raise ValueError("Valid split values are 'train' and 'test'")

        self.df = self.df[self.df.has_midi & self.df.has_nlm & self.df.has_music_track_description]

        #TODO create dataset with and without descriptions
        #TODO use without descriptions for CFG

        self.df = self.df.reset_index()

        logger.info(f"Initialized dataset from csv.\nHead:\n{self.df[['file_name_prefix', 'split']].head()}\nSize:{len(self)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # row = self.metadata.loc[index][self.metadata_filter].to_dict()
        row = self.df.loc[index].to_dict()
        file_name_prefix = row["file_name_prefix"]
        wav_path = os.path.join(self.dataset_root, row["file_path_wav"])
        desc_file_path = os.path.join(self.dataset_root, row["file_path_music_track_description"])

        desc_tags = [f"game: {row['file_game_name']}", f"track name: {row['track_name']}", f"game genre: {str(row['genre_wiki']).strip('[]')}", f"{row['genre_cluster']}", f"bpm: {row['tempo_bpm']}"]
        random.shuffle(desc_tags)
        desc_tags = ", ".join(desc_tags)

        # Get text tokens
        with open(desc_file_path, 'r') as f:
            file_txt = f.read()
            with torch.no_grad():
                description = desc_tags + file_txt.strip()
                logger.debug(f"Audio description: {description}")

        return {
            'file_name_prefix': file_name_prefix,
            'description': description, 
            'wav_path': wav_path
        }

