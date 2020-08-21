import soundfile as sf
import librosa
import numpy as np
from config.config import opt
from python_speech_features import mfcc, logfbank, delta
from scipy.signal.windows import hamming


def load_audio(filename, offset=0, duration=None, resample=True):
    # 读取语音数据，filename是文件地址，start，stop是这条语音开始读取位置，stop是结束位置
    # y = None
    # sr = opt.SAMPLE_RATE
    # try:
    #     y, sr = sf.read(filename, start=start, stop=stop, dtype='float32', always_2d=True)
    # except Exception as e:
    #     print(e)
    #     print(filename)
    # else:
    #     y = y[:, 0]
    #     if resample and sr != opt.SAMPLE_RATE:  # 如果采样率不是16000，进行重采样
    #         y = librosa.resample(y, sr, opt.SAMPLE_RATE)
    #         return y, opt.SAMPLE_RATE
    # finally:
    #     return y, sr
    y = None
    try:
        y, sr = librosa.load(filename, offset=offset, duration=duration)
    except Exception as e:
        print(e)
        print(filename)
    else:
        y = y[:, 0]
        if resample and sr != opt.SAMPLE_RATE:
            y = librosa.resample(y, sr, opt.SAMPLE_RATE)
            return y, opt.SAMPLE_RATE
    finally:
        return y, sr


def make_feature(y, sr=opt.SAMPLE_RATE):
    N_FFT = int(opt.WIN_LEN * opt.SAMPLE_RATE)  # 滑窗采样点
    HOP_LEN = int(opt.WIN_STEP * opt.SAMPLE_RATE)  # 滑窗滑动距离采样点
    # 提取特征，y是语音data部分，sr为采样率
    if opt.FEATURE == 'fft':  # 提取fft特征
        S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LEN, window=hamming)  # 进行短时傅里叶变换
        feature, _ = librosa.magphase(S)
        feature = np.log1p(feature)  # log1p操作
        feature = feature.transpose()
    else:
        if opt.FEATURE == 'logfbank':  # 提取fbank特征
            feature = logfbank(y, sr, winlen=opt.WIN_LEN, winstep=opt.WIN_STEP)
        else:
            feature = mfcc(y, sr, winlen=opt.WIN_LEN, winstep=opt.WIN_STEP)  # 提取mfcc特征
        feature_d1 = delta(feature, N=1)  # 加上两个delta，特征维度X3
        feature_d2 = delta(feature, N=2)
        feature = np.hstack([feature, feature_d1, feature_d2])  # 横向拼起来
    return normalize(feature)  # 返回归一化的特征


def normalize(v):  # 进行归一化，v是语音特征
    return (v - v.mean(axis=0)) / (v.std(axis=0) + 2e-12)


def mono_to_color(
    X: np.ndarray, mean=None, std=None,
    norm_max=None, norm_min=None, eps=1e-6
):
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V


if __name__ == "__main__":
    y, sr = load_audio('D:/project/Cornell-Birdcall-Identification/data/birdsong-recognition/train_audio/aldfly/XC31060.mp3')
    spec = make_feature(y, sr)
    print(spec.shape)
