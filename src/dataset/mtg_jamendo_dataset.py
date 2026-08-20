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

        splits = [split] if split == "test" else [split, "validation"] 

        self.files_paths = []
        for split in splits:
            files_folder = os.path.join(dataset_root, split)

            for file in os.listdir(files_folder):
                file_path = os.path.join(files_folder, file)
                self.files_paths.append(file_path)

    def __len__(self):
        return len(self.files_paths)

    def __getitem__(self, index):
        file_path = self.files_paths[index]
        pre_comp = torch.load(file_path, map_location='cpu')
        pre_comp['file_name_prefix'] = file_path.split('/')[-1][:-3]

        return pre_comp
