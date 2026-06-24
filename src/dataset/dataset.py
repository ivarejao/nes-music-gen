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
        pre_comp_path = wav_path[:-4] + '_pre_comp.pt'

        pre_comp = torch.load(pre_comp_path, map_location='cpu')
        pre_comp['file_name_prefix'] = file_name_prefix

        return pre_comp

