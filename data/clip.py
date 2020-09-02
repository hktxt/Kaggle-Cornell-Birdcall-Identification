import soundfile as sf
import os
import numpy as np


root_pth = 'D:/project/Cornell-Birdcall-Identification/data/birdsong-recognition/train_audio_resampled/'
save_pth = 'D:/project/Cornell-Birdcall-Identification/data/birdsong-recognition/resampled_clips/'
if not os.path.exists(save_pth):
    os.makedirs(save_pth)
bird_folds = os.listdir(root_pth)

############################################
duration = 5              # 5s duration
step_size = 2.5           # the step overlap
ignore = 2                # wav less then 2s will be skipped
sample_rate = 32000
N_SAMPLES = duration * sample_rate
############################################

for bird in bird_folds:
    bird_pth = os.path.join(root_pth, bird)
    wav_files = os.listdir(bird_pth)
    save_fold = os.path.join(save_pth, bird)
    if not os.path.exists(save_fold):
        os.makedirs(save_fold)
    for wav_file in wav_files:
        wav_pth = os.path.join(bird_pth, wav_file)
        print(wav_pth)
        y, sr = sf.read(wav_pth)
        assert sr == sample_rate, 'sr Err.'
        length_y = len(y)
        if length_y < ignore * sample_rate:
            continue
        if length_y < N_SAMPLES:
            new_y = np.zeros(N_SAMPLES, dtype=y.dtype)
            start = np.random.randint(N_SAMPLES - length_y)
            new_y[start:start + length_y] = y
            y = new_y.astype(np.float32)
            # Write out audio
            f = os.path.join(save_fold, wav_file)
            sf.write(f, y, sample_rate)
        elif length_y == N_SAMPLES:
            f = os.path.join(save_fold, wav_file)
            sf.write(f, y, sample_rate)
        else:
            i = 0
            while True:
                start = i * step_size * sample_rate
                end = start + N_SAMPLES
                if start < length_y:
                    if end < length_y:
                        clip = y[int(start):int(end)]
                    else:
                        break
                else:
                    break
                i += 1
                f = os.path.join(save_fold, wav_file.replace('.wav', f'_{i}.wav'))
                sf.write(f, clip, sample_rate)

