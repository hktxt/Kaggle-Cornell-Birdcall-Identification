# organize data, read wav to get duration and split train/test
# to a csv file
# author: Max, 2020.09.02

import os
import librosa
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def main(root_pth):
    if not os.path.exists('df_clips.csv'):
        data = []
        folds = os.listdir(root_pth)
        for idx, fold in enumerate(tqdm(folds)):
            wavs = os.listdir(os.path.join(root_pth, fold))
            for wav in wavs:
                # wav_pth = os.path.join(root_pth, fold, wav)
                # duration = librosa.get_duration(filename=wav_pth)
                duration = 5

                target = {
                    'resampled_filename': wav,
                    'xc_id': wav.split('.')[0],
                    'ebird_code': fold,
                    'label': idx,
                    'duration': duration
                }
                data.append(target)

            # if idx == 1:
            #     break

        df = pd.DataFrame(data, columns=['label', 'ebird_code', 'resampled_filename', 'xc_id', 'duration'])
        df.to_csv('df_clips.csv', index=False)

    df = pd.read_csv('df_clips.csv')
    # df['fold'] = -1
    # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # for fold_number, (train_index, val_index) in enumerate(skf.split(X=df.index, y=df['label'])):
    #     df.loc[df.iloc[val_index].index, 'fold'] = fold_number
    # df.to_csv('df_clips.csv', index=False)


if __name__ == "__main__":
    main('D:/project/Cornell-Birdcall-Identification/data/birdsong-recognition/resampled_clips/')

    #  https://www.kaggle.com/ttahara/training-birdsong-baseline-resnest50-fast#split-data
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_all = pd.read_csv('df_clips.csv')
    train_all["fold"] = -1
    for fold_id, (train_index, val_index) in enumerate(skf.split(train_all, train_all["ebird_code"])):
        train_all.iloc[val_index, -1] = fold_id

    fold_proportion = pd.pivot_table(train_all, index="ebird_code", columns="fold", values="xc_id", aggfunc=len)
    print(fold_proportion.shape)

    train_all.to_csv('df_clips_mod.csv', index=False)
