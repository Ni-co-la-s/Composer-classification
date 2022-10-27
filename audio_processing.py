import librosa.display
import numpy as np
import librosa
import os
from tqdm import tqdm
import random
import re
import pickle
from utils import (
    rootdir_train_extract,
    rootdir_test_extract,
    RATE,
    N_MFCC,
    segment_length,
    hop_length,
    N_CHROMA,
    AudioAugmentation,
    L_composer,
    PATH_PICKLE_FOLDER,
)


# Create the training and validation dataset by concatenating the MFCC and the
# CHROMA computed with librosa.
# The training data are augmented (noise, shift, small transposition of pitch)


aa = AudioAugmentation()


for subdir, dirs, files in os.walk(rootdir_train_extract):
    for k in range(8, (len(files) // segment_length)):

        print(k)
        X = []
        labels = []
        X2 = []
        labels2 = []

        for file in tqdm(files[k * segment_length : (k + 1) * segment_length]):

            p = re.compile("(.*)_[0-9]*_[0-9]*")
            result = p.search(file)

            if result == None:
                continue
            label = result.group(1)

            if L_composer != []:
                if label not in L_composer:
                    continue
            try:
                y, sr = librosa.load(os.path.join(subdir, file), sr=RATE)
            except:

                continue
            number_data_generation = random.randint(5)

            # Augment training data

            for i in range(number_data_generation):

                mfcc = np.array(
                    librosa.feature.mfcc(
                        y=y, sr=sr, n_mfcc=N_MFCC, hop_length=hop_length
                    )
                )
                chroma = np.array(
                    librosa.feature.chroma_stft(
                        y=y, sr=sr, hop_length=hop_length, n_chroma=N_CHROMA
                    )
                )
                feature = np.concatenate((mfcc, chroma))

                r = 0.8 + random.random() * 0.4

                r = random.random() * 0.1

                y = aa.add_noise(y, r)

                r = random.randint(-6, 6)

                y = aa.pitch(y, sr, r)
                y = aa.shift(y)

                X.append(feature)
                labels.append(label)
        X = np.array(X)

        file_to_store = open(
            os.path.join(PATH_PICKLE_FOLDER, "Dataset_train_test" + str(k) + ".pkl"),
            "wb",
        )

        pickle.dump([X, labels], file_to_store)

        file_to_store.close()
for subdir, dirs, files in os.walk(rootdir_test_extract):
    for k in range(0, (len(files) // segment_length)):

        print(k)
        X = []
        labels = []
        for file in tqdm(
            files[k * segment_length : (k + 1) * segment_length]
        ):  # 18885:

            p = re.compile("(.*)_[0-9]*_[0-9]*")
            result = p.search(file)

            if result == None:
                continue
            label = result.group(1)

            if L_composer != []:
                if label not in L_composer:
                    continue
            try:
                y, sr = librosa.load(os.path.join(subdir, file), sr=RATE)
            except:

                continue
            mfcc = np.array(
                librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=hop_length)
            )
            chroma = np.array(
                librosa.feature.chroma_stft(
                    y=y, sr=sr, hop_length=hop_length, n_chroma=N_MFCC
                )
            )
            feature = np.concatenate((mfcc, chroma))

            X.append(feature)
            labels.append(label)
        X = np.array(X)

        file_to_store = open(
            os.path.join(PATH_PICKLE_FOLDER, "Dataset_validation" + str(k) + ".pkl"),
            "wb",
        )
        pickle.dump([X, labels], file_to_store)

        file_to_store.close()
