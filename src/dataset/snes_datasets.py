import os
import logging
import json

import pandas as pd

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class NESDatasetBase(Dataset):
    def __init__(self, dataset_root:str, split:str) -> None:
        super().__init__()

        self.dataset_root = dataset_root
        self.csv_path = os.path.join(self.dataset_root, "nes_metadata.csv")

        self.df = pd.read_csv(self.csv_path, dtype={"id":"object"})

        if split == "train":
            self.df:pd.DataFrame = self.df[self.df.split.isin(["train", "valid"])]
        elif split == "test":
            self.df:pd.DataFrame = self.df[self.df["split"] == "test"]
        else:
            raise ValueError("Valid split values are 'train' and 'test'")

        self.df = self.df[self.df.has_midi & self.df.has_nlm & self.df.has_music_track_description]
        self.df = self.df.reset_index()

        logger.info(f"Initialized dataset from csv.\nHead:\n{self.df[['file_name_prefix', 'split']].head()}\nSize:{len(self)}")

    def __len__(self):
        return len(self.df)


class NESTestTokenizersDataset(NESDatasetBase):
    def __init__(self, dataset_root:str, split:str) -> None:
        super().__init__(dataset_root=dataset_root, split=split)

    def __getitem__(self, index):
        # row = self.metadata.loc[index][self.metadata_filter].to_dict()
        row = self.df.loc[index].to_dict()
        file_name_prefix = row["file_name_prefix"]
        wav_path = os.path.join(self.dataset_root, row["file_path_wav"])

        return file_name_prefix, wav_path


class NESMusicGenBaseDataset(NESDatasetBase):
    def __init__(self, dataset_root:str, split:str) -> None:
        super().__init__(dataset_root=dataset_root, split=split)

    def __getitem__(self, index):
        # row = self.metadata.loc[index][self.metadata_filter].to_dict()
        row = self.df.loc[index].to_dict()
        file_name_prefix = row["file_name_prefix"]
        desc_file_path = os.path.join(self.dataset_root, row["file_path_music_track_description"])

        # Get text tokens
        with open(desc_file_path, 'r') as f:
            desc_file_txt = f.read()

        return file_name_prefix, desc_file_txt


class NESAudioDataset(NESDatasetBase):
    def __init__(self, dataset_root:str, split:str, num_codebooks, audio_secs:int) -> None:
        super().__init__(dataset_root=dataset_root, split=split)

        self.num_codebooks = num_codebooks
        self.audio_secs = audio_secs

    def __getitem__(self, index):
        # row = self.metadata.loc[index][self.metadata_filter].to_dict()
        row = self.df.loc[index].to_dict()
        file_name_prefix = row["file_name_prefix"]
        wav_path = os.path.join(self.dataset_root, row["file_path_wav"])
        pre_comp_path = wav_path[:-4] + '_pre_comp.pt'

        pre_comp = torch.load(pre_comp_path, map_location='cpu')
        pre_comp['file_name_prefix'] = file_name_prefix

        return pre_comp


MIDI_BOS_TKN = 1
MIDI_EOS_TKN = 2

class NESMidiDataset(NESDatasetBase):
    def __init__(self, dataset_root:str, split:str) -> None:
        super().__init__(dataset_root=dataset_root, split=split)

    def __getitem__(self, index):
        # row = self.metadata.loc[index][self.metadata_filter].to_dict()
        row = self.df.loc[index].to_dict()
        item_dict = {}

        file_name_prefix = row["file_name_prefix"]
        item_dict['file_name_prefix'] = file_name_prefix

        # Get MIDI tokens
        midi_path = os.path.join(self.dataset_root, row['file_path_midi_tokens'])
        with open(midi_path, 'r') as f:
            midi_tokens = json.load(f)

        if len(midi_tokens) > 6_000:
            midi_tokens = [MIDI_BOS_TKN] + midi_tokens[:6_000]
        else:
            midi_tokens = [MIDI_BOS_TKN] + midi_tokens + [MIDI_EOS_TKN]

        midi_tokens = torch.tensor(midi_tokens, dtype=torch.long)

        input_tokens = midi_tokens[:-1]
        target_tokens = midi_tokens[1:]

        item_dict['input_tokens'] = input_tokens
        item_dict['target_tokens'] = target_tokens

        # Get description T5 embeddings
        wav_path = os.path.join(self.dataset_root, row["file_path_wav"])
        pre_comp_path = wav_path[:-4] + '_pre_comp.pt'
        pre_comp = torch.load(pre_comp_path, map_location='cpu')

        item_dict['txt_tokens'] = pre_comp['txt_tokens']
        item_dict['txt_tokens_mask'] = pre_comp['txt_tokens_mask']

        return item_dict


NLM_BOS_TKN = 1
NLM_EOS_TKN = 2

class NESNlmDataset(NESDatasetBase):
    def __init__(self, dataset_root:str, split:str) -> None:
        super().__init__(dataset_root=dataset_root, split=split)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # row = self.metadata.loc[index][self.metadata_filter].to_dict()
        row = self.df.loc[index].to_dict()
        item_dict = {}

        file_name_prefix = row["file_name_prefix"]
        item_dict['file_name_prefix'] = file_name_prefix

        # Get NLM tokens
        nlm_path = os.path.join(self.dataset_root, row['file_path_nlm_tokens'])
        with open(nlm_path, 'r') as f:
            nlm_tokens = json.load(f)

        if len(nlm_tokens) > 6_000:
            nlm_tokens = [NLM_BOS_TKN] + nlm_tokens[:6_000]
        else:
            nlm_tokens = [NLM_BOS_TKN] + nlm_tokens + [NLM_EOS_TKN]

        nlm_tokens = torch.tensor(nlm_tokens, dtype=torch.long)

        input_tokens = nlm_tokens[:-1]
        target_tokens = nlm_tokens[1:]

        item_dict['input_tokens'] = input_tokens
        item_dict['target_tokens'] = target_tokens

        # Get description T5 embeddings
        wav_path = os.path.join(self.dataset_root, row["file_path_wav"])
        pre_comp_path = wav_path[:-4] + '_pre_comp.pt'
        pre_comp = torch.load(pre_comp_path, map_location='cpu')

        item_dict['txt_tokens'] = pre_comp['txt_tokens']
        item_dict['txt_tokens_mask'] = pre_comp['txt_tokens_mask']

        return item_dict