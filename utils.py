import os
import librosa
import numpy as np
import soundfile as sf

# Constants

PATH_ORIGINAL_CSV = "maestro-v2.0.0.csv"
PATH_MODIF_CSV = "maestro.csv"
PATH_PICKLE_FOLDER = ""
rootdir_songs = "songs"
rootdir_test_extract = "test_songs"
rootdir_train_extract = "train_songs"
duration_extract = 20
RATE = 22000
N_MFCC = 12
N_CHROMA = 12
segment_length = 1000
hop_length = 1024
L_composer = [
    "Ludwig van Beethoven",
    "Franz Schubert",
    "Franz Liszt",
    "Frédéric Chopin",
]


# Functions


def add_file(rootdir, filename, extension):
    """
    Add file to the rootdir, if duplicate, add a number at the end of the filename
    """
    if not os.path.isfile(os.path.join(rootdir, filename)):

        b = True
        a = 1
        while b == True:
            renamed_file = filename + "_" + str(a) + extension
            if not os.path.isfile(os.path.join(rootdir, renamed_file)):
                b = False
            a = a + 1
        return os.path.join(rootdir, renamed_file)
    else:
        return os.path.join(rootdir, filename)


class AudioAugmentation:
    """
    Handles the augmentation of audio files
    """

    def read_audio_file(self, file_path):
        data = librosa.core.load(file_path)[0]
        return data

    def write_audio_file(self, file, data, sample_rate=16000):
        sf.write(file, data, sample_rate)

    def add_noise(self, data, noise_factor=0.005):
        noise = np.random.randn(len(data))
        data_noise = data + noise_factor * noise
        return data_noise

    def shift(self, data):
        shift = np.random.randint(len(data) / 3, 2 * len(data) / 3)
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift
        augmented_data = np.roll(data, shift)
        if shift > 0:
            augmented_data[:shift] = 0
        else:
            augmented_data[shift:] = 0
        return augmented_data

    def stretch(self, data, rate=1):
        data = librosa.effects.time_stretch(data, rate=rate)

        return data

    def pitch(self, data, sr, n_steps=0):
        return librosa.effects.pitch_shift(data, sr=sr, n_steps=n_steps)
