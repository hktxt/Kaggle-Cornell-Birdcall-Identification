import os
import webrtcvad
from dataset.vad import read_wave, frame_generator, vad_collector, write_wave


def main(mode=0):
    """this script filters out non-voiced audio frames."""

    pth = 'D:/project/Cornell-Birdcall-Identification/data/birdsong-recognition/train_audio_resampled/'
    save_pth = 'D:/project/Cornell-Birdcall-Identification/data/birdsong-recognition/VAD/'

    if not os.path.exists(save_pth):
        os.makedirs(save_pth)
    birds = os.listdir(pth)

    vad = webrtcvad.Vad(mode)
    for bird in birds:
        bird_pth = os.path.join(pth, bird)
        wavs = os.listdir(bird_pth)
        save_fold = os.path.join(save_pth, bird)
        if not os.path.exists(save_fold):
            os.makedirs(save_fold)
        for wav in wavs:
            wav_pth = os.path.join(bird_pth, wav)
            audio, sample_rate = read_wave(wav_pth)
            frames = frame_generator(30, audio, sample_rate)
            frames = list(frames)
            segments = vad_collector(sample_rate, 30, 300, vad, frames)
            concat_segments = b''
            for i, segment in enumerate(segments):
                concat_segments = b"".join([concat_segments, segment])
            path = os.path.join(save_fold, wav)
            write_wave(path, concat_segments, sample_rate)

    print('Done!')


if __name__ == "__main__":
    # change the path.
    # set mode: 0, 1, 2, 3
    main(0)
