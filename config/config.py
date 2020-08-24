

class Config:

    # feature config
    SAMPLE_RATE = 32000
    FEATURE = 'fft'    # 特征类型, fft/logfbank
    FEATURE_LEN = 161  # 特征维度
    WIN_LEN = 0.025     # 滑窗窗口长度，单位s
    WIN_STEP = 0.01     # 滑窗滑动距离，单位s
    N_FFT = int(WIN_LEN * SAMPLE_RATE)     # 滑窗采样点
    HOP_LEN = int(WIN_STEP * SAMPLE_RATE)  # 滑窗滑动距离采样点

    N_FRAMES = 500     # 训练帧数
    DURATION = N_FRAMES * WIN_STEP     # 训练单句时长
    N_SAMPLES = int(DURATION * SAMPLE_RATE)  # 训练单句采样点

    ROOT_PTH = 'D:/project/Cornell-Birdcall-Identification/data/birdsong-recognition/train_audio_resampled/'

    VAD = False

opt = Config()