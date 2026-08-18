import os
import logging

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class MTGJamendoAudioDataset(Dataset):
    def __init__(self, dataset_root:str, split:str, num_codebooks:int, audio_secs:int) -> None:
        super().__init__()

        self.num_codebooks = num_codebooks
        self.audio_secs = audio_secs

        self.files_folder = os.path.join(dataset_root, split)
        self.files = os.listdir(self.files_folder)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file = self.files[index]
        pre_comp_path = os.path.join(self.files_folder, file)
        pre_comp = torch.load(pre_comp_path, map_location='cpu')
        pre_comp['file_name_prefix'] = file.split('/')[0]

        return pre_comp
