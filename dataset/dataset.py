# author: Max, 2020.08.5
from torch.utils.data import Dataset
import os
from .utils import load_audio, make_feature
from config.config import opt
from .utils import mono_to_color
import cv2
import random
import librosa
import soundfile as sf
import numpy as np


BIRD_CODE = {
    'aldfly': 0, 'ameavo': 1, 'amebit': 2, 'amecro': 3, 'amegfi': 4,
    'amekes': 5, 'amepip': 6, 'amered': 7, 'amerob': 8, 'amewig': 9,
    'amewoo': 10, 'amtspa': 11, 'annhum': 12, 'astfly': 13, 'baisan': 14,
    'baleag': 15, 'balori': 16, 'banswa': 17, 'barswa': 18, 'bawwar': 19,
    'belkin1': 20, 'belspa2': 21, 'bewwre': 22, 'bkbcuc': 23, 'bkbmag1': 24,
    'bkbwar': 25, 'bkcchi': 26, 'bkchum': 27, 'bkhgro': 28, 'bkpwar': 29,
    'bktspa': 30, 'blkpho': 31, 'blugrb1': 32, 'blujay': 33, 'bnhcow': 34,
    'boboli': 35, 'bongul': 36, 'brdowl': 37, 'brebla': 38, 'brespa': 39,
    'brncre': 40, 'brnthr': 41, 'brthum': 42, 'brwhaw': 43, 'btbwar': 44,
    'btnwar': 45, 'btywar': 46, 'buffle': 47, 'buggna': 48, 'buhvir': 49,
    'bulori': 50, 'bushti': 51, 'buwtea': 52, 'buwwar': 53, 'cacwre': 54,
    'calgul': 55, 'calqua': 56, 'camwar': 57, 'cangoo': 58, 'canwar': 59,
    'canwre': 60, 'carwre': 61, 'casfin': 62, 'caster1': 63, 'casvir': 64,
    'cedwax': 65, 'chispa': 66, 'chiswi': 67, 'chswar': 68, 'chukar': 69,
    'clanut': 70, 'cliswa': 71, 'comgol': 72, 'comgra': 73, 'comloo': 74,
    'commer': 75, 'comnig': 76, 'comrav': 77, 'comred': 78, 'comter': 79,
    'comyel': 80, 'coohaw': 81, 'coshum': 82, 'cowscj1': 83, 'daejun': 84,
    'doccor': 85, 'dowwoo': 86, 'dusfly': 87, 'eargre': 88, 'easblu': 89,
    'easkin': 90, 'easmea': 91, 'easpho': 92, 'eastow': 93, 'eawpew': 94,
    'eucdov': 95, 'eursta': 96, 'evegro': 97, 'fiespa': 98, 'fiscro': 99,
    'foxspa': 100, 'gadwal': 101, 'gcrfin': 102, 'gnttow': 103, 'gnwtea': 104,
    'gockin': 105, 'gocspa': 106, 'goleag': 107, 'grbher3': 108, 'grcfly': 109,
    'greegr': 110, 'greroa': 111, 'greyel': 112, 'grhowl': 113, 'grnher': 114,
    'grtgra': 115, 'grycat': 116, 'gryfly': 117, 'haiwoo': 118, 'hamfly': 119,
    'hergul': 120, 'herthr': 121, 'hoomer': 122, 'hoowar': 123, 'horgre': 124,
    'horlar': 125, 'houfin': 126, 'houspa': 127, 'houwre': 128, 'indbun': 129,
    'juntit1': 130, 'killde': 131, 'labwoo': 132, 'larspa': 133, 'lazbun': 134,
    'leabit': 135, 'leafly': 136, 'leasan': 137, 'lecthr': 138, 'lesgol': 139,
    'lesnig': 140, 'lesyel': 141, 'lewwoo': 142, 'linspa': 143, 'lobcur': 144,
    'lobdow': 145, 'logshr': 146, 'lotduc': 147, 'louwat': 148, 'macwar': 149,
    'magwar': 150, 'mallar3': 151, 'marwre': 152, 'merlin': 153, 'moublu': 154,
    'mouchi': 155, 'moudov': 156, 'norcar': 157, 'norfli': 158, 'norhar2': 159,
    'normoc': 160, 'norpar': 161, 'norpin': 162, 'norsho': 163, 'norwat': 164,
    'nrwswa': 165, 'nutwoo': 166, 'olsfly': 167, 'orcwar': 168, 'osprey': 169,
    'ovenbi1': 170, 'palwar': 171, 'pasfly': 172, 'pecsan': 173, 'perfal': 174,
    'phaino': 175, 'pibgre': 176, 'pilwoo': 177, 'pingro': 178, 'pinjay': 179,
    'pinsis': 180, 'pinwar': 181, 'plsvir': 182, 'prawar': 183, 'purfin': 184,
    'pygnut': 185, 'rebmer': 186, 'rebnut': 187, 'rebsap': 188, 'rebwoo': 189,
    'redcro': 190, 'redhea': 191, 'reevir1': 192, 'renpha': 193, 'reshaw': 194,
    'rethaw': 195, 'rewbla': 196, 'ribgul': 197, 'rinduc': 198, 'robgro': 199,
    'rocpig': 200, 'rocwre': 201, 'rthhum': 202, 'ruckin': 203, 'rudduc': 204,
    'rufgro': 205, 'rufhum': 206, 'rusbla': 207, 'sagspa1': 208, 'sagthr': 209,
    'savspa': 210, 'saypho': 211, 'scatan': 212, 'scoori': 213, 'semplo': 214,
    'semsan': 215, 'sheowl': 216, 'shshaw': 217, 'snobun': 218, 'snogoo': 219,
    'solsan': 220, 'sonspa': 221, 'sora': 222, 'sposan': 223, 'spotow': 224,
    'stejay': 225, 'swahaw': 226, 'swaspa': 227, 'swathr': 228, 'treswa': 229,
    'truswa': 230, 'tuftit': 231, 'tunswa': 232, 'veery': 233, 'vesspa': 234,
    'vigswa': 235, 'warvir': 236, 'wesblu': 237, 'wesgre': 238, 'weskin': 239,
    'wesmea': 240, 'wessan': 241, 'westan': 242, 'wewpew': 243, 'whbnut': 244,
    'whcspa': 245, 'whfibi': 246, 'whtspa': 247, 'whtswi': 248, 'wilfly': 249,
    'wilsni1': 250, 'wiltur': 251, 'winwre3': 252, 'wlswar': 253, 'wooduc': 254,
    'wooscj2': 255, 'woothr': 256, 'y00475': 257, 'yebfly': 258, 'yebsap': 259,
    'yehbla': 260, 'yelwar': 261, 'yerwar': 262, 'yetvir': 263
}

INV_BIRD_CODE = {v: k for k, v in BIRD_CODE.items()}


class Birdcall(Dataset):
    def __init__(self, df, pth=opt.ROOT_PTH, train=True, transform=None):
        self.df = df
        self.pth = pth
        self.train = train
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        info = []
        sample = self.df.iloc[idx]
        wav_name = sample["resampled_filename"]
        ebird_code = sample["ebird_code"]
        wav_pth = os.path.join(self.pth, ebird_code, wav_name)
        # y, _ = load_audio(wav_pth)
        y, _ = librosa.load(wav_pth)
        label = BIRD_CODE[ebird_code]
        if self.train:
            n_samples = len(y)
            y = [y]
            while n_samples <= opt.N_SAMPLES:
                sub_df = self.df[self.df['ebird_code'] == ebird_code]
                aid = random.randrange(0, len(sub_df))
                _sample = sub_df.iloc[aid]
                _wav_name = _sample["resampled_filename"]
                _ebird_code = _sample["ebird_code"]
                assert _ebird_code == ebird_code, "Err."
                _wav_pth = os.path.join(self.pth, _ebird_code, _wav_name)
                _y, _ = librosa.load(_wav_pth)
                y.append(_y)
                n_samples += len(_y)

            # random sample
            upper_bound = n_samples - opt.N_SAMPLES
            start = np.random.randint(0, upper_bound)
            end = start + opt.N_SAMPLES
            spec = np.array([make_feature(np.hstack(y)[start: end], opt.SAMPLE_RATE).transpose()])
        else:
            spec = np.array([make_feature(y, opt.SAMPLE_RATE).transpose()])
        spec = spec.transpose(1, 2, 0)

        sample = {
            'image': spec,
            'label': label
        }

        if self.transform:
            spec = self.transform(**sample)['image']
            label = self.transform(**sample)['label']

        return spec, label, info


def callback_get_label(dataset, idx):
    return int(dataset[idx][1])


def callback_get_label1(dataset, idx):
    return int(dataset[idx][1][dataset[idx][1]==1])


# modified from: https://www.kaggle.com/ttahara/training-birdsong-baseline-resnest50-fast
class SpectrogramDataset(Dataset):
    def __init__(
        self,
        df, pth=opt.ROOT_PTH, img_size=224,
        waveform_transforms=None, spectrogram_transforms=None, melspectrogram_parameters={}, PERIOD=5
    ):
        self.df = df
        self.pth = pth
        self.img_size = img_size
        self.waveform_transforms = waveform_transforms
        self.spectrogram_transforms = spectrogram_transforms
        self.melspectrogram_parameters = melspectrogram_parameters
        self.PERIOD = PERIOD

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        sample = self.df.iloc[idx]
        wav_name = sample["resampled_filename"]
        ebird_code = sample["ebird_code"]
        wav_pth = os.path.join(self.pth, ebird_code, wav_name)

        y, sr = sf.read(wav_pth)

        if self.waveform_transforms:
            y = self.waveform_transforms(y)
        else:
            len_y = len(y)
            effective_length = sr * self.PERIOD
            # 采样不足时，拼接相同鸟的特征，而不是补0.
            samples = [y]
            if len_y < effective_length:
                while len_y <= effective_length:
                    sub_df = self.df[self.df['ebird_code'] == ebird_code]
                    aid = random.randrange(0, len(sub_df))
                    _sample = sub_df.iloc[aid]
                    _wav_name = _sample["resampled_filename"]
                    _ebird_code = _sample["ebird_code"]
                    assert _ebird_code == ebird_code, "Err."
                    _wav_pth = os.path.join(self.pth, _ebird_code, _wav_name)
                    new_y, _ = sf.read(_wav_pth)
                    samples.append(new_y)
                    len_y += len(new_y)
                new_y = np.hstack(samples)
                start = np.random.randint(len_y - effective_length)
                y = new_y[start:start + effective_length]
                y = y.astype(np.float32)
            elif len_y > effective_length:
                start = np.random.randint(len_y - effective_length)
                y = y[start:start + effective_length].astype(np.float32)
            else:
                y = y.astype(np.float32)

        melspec = librosa.feature.melspectrogram(y, sr=sr, **self.melspectrogram_parameters)
        melspec = librosa.power_to_db(melspec).astype(np.float32)

        if self.spectrogram_transforms:
            melspec = self.spectrogram_transforms(melspec)

        image = mono_to_color(melspec)
        height, width, _ = image.shape
        image = cv2.resize(image, (int(width * self.img_size / height), self.img_size))
        image = np.moveaxis(image, 2, 0)
        image = (image / 255.0).astype(np.float32)

#         labels = np.zeros(len(BIRD_CODE), dtype="i")
        labels = np.zeros(len(BIRD_CODE), dtype="f")
        labels[BIRD_CODE[ebird_code]] = 1

        return image, labels  #, BIRD_CODE[ebird_code]


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv('../data/df.csv')
    df = df[df['fold'] == 0]
    data = Birdcall(df)
    spec, label, info = data[1]
    import VisionToolKit as vtk
    vtk.imshow(spec[0, :, :])
    print(spec)