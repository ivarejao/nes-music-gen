import os
import pandas as pd

DATASET_ROOT = "/home/es119256/dados/datasets/nes_dataset"

csv_path = os.path.join(DATASET_ROOT, "nes_metadata.csv")
df = pd.read_csv(csv_path, dtype={"id":"object"})
df:pd.DataFrame = df[df.has_midi & df.has_nlm]
df = df.where(df.notnull(), None)

index_count = 0
safe_dict = {}

count_wav = 0
count_desc = 0
count_split = 0

for row in df.to_dict(orient='records'):
    wav_path = os.path.join(DATASET_ROOT, row["file_path_wav"])
    file_path_desc = wav_path[:-4]+'_music_track_description.md'

    no_wav = not os.path.exists(wav_path)
    no_desc = not row["file_path_music_track_description"]
    no_split = not row["split"]

    print(row["file_path_music_track_description"])

    if (not no_wav) and (not no_desc):
        safe_dict[index_count] = row
        safe_dict[index_count]["file_path_desc"] = file_path_desc
        index_count += 1

    if no_wav or no_desc:
        if no_wav:
            print(f"NO WAV {wav_path}")
            count_desc += 1
            count_wav += 1

        if no_desc and not no_wav:
            print(f"NO DESC FOR {wav_path}")
            count_desc += 1

    if no_split:
        count_split += 1

print(f"no_wav: {count_wav}, no_desc: {count_desc}, no_split: {count_split}")