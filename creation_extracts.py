import csv
from tqdm import tqdm
import os
from pathlib import Path
import subprocess
import random
import pickle
from pydub import AudioSegment
from utils import (
    PATH_MODIF_CSV,
    rootdir_train_extract,
    rootdir_songs,
    rootdir_test_extract,
    duration_extract,
    add_file,
    PATH_PICKLE_FOLDER,
)


# Create the extracts of the pieces retrieved and separate them between
# training and validation


dic_pieces = {}  # For each file, indicate the composer
duc_title_mp3 = {}  # For each original mp3 file, indicate the title
dic_title_wav = {}  # For each final wav file, indicate the title
dic_composer = {}  # For each composer, indicate the number of pieces treated


with open(PATH_MODIF_CSV, "r", encoding="utf8") as csvfile:
    datareader = csv.reader(csvfile, delimiter=",",)
    first_line = False
    for row in tqdm(datareader):
        if not (first_line):
            first_line = True
            continue
        composer = row[1]
        filename = row[6][5:-4] + ".mp3"
        title = row[2]
        dic_pieces[filename] = composer
        duc_title_mp3[filename] = title
        if composer not in dic_composer:
            dic_composer[composer] = 0
for subdir, dirs, files in os.walk(rootdir_songs):

    for file in tqdm(files):

        if Path(os.path.join(subdir, file)).suffix == ".mp3":

            if file not in dic_pieces:
                continue
            composer = dic_pieces[file]
            title = duc_title_mp3[file]

            title = title.replace('"', "")  # Takes care of problems of naming

            title = title.replace(":", "")

            n = dic_composer[composer]
            dic_composer[composer] += 1

            r = random.random()

            if r < 0.2:  # 20 percent of the pieces are used for validation
                rootdir_piece = rootdir_test_extract
            else:
                rootdir_piece = rootdir_train_extract
            wav_file = composer + "_" + str(dic_composer[composer]) + ".wav"

            dic_title_wav[wav_file] = title

            subprocess.call(
                [
                    "ffmpeg",
                    "-i",
                    (os.path.join(subdir, file)),
                    (os.path.join(rootdir_piece, wav_file)),
                ]
            )

            piece = AudioSegment.from_wav((os.path.join(rootdir_piece, wav_file)))
            duration = int(piece.duration_seconds)

            # We create extracts of 20 seconds of the pieces

            for i in range(duration // duration_extract + 2):

                r_sec = random.randint(0, duration - duration_extract)

                StrtMin = r_sec // 60
                StrtSec = r_sec % 60
                EndMin = (r_sec + duration_extract) % 60
                EndSec = (r_sec + duration_extract) % 60

                # Time to milliseconds conversion
                StrtTime = StrtMin * 60 * 1000 + StrtSec * 1000
                EndTime = StrtMin * 60 * 1000 + EndSec * 1000
                # Opening file and extracting portion of it
                extract = piece[StrtTime:EndTime]
                # Saving file in required location
                renamed_file = add_file(rootdir_piece, wav_file[:-4], ".wav")

                extract.export((os.path.join(rootdir_piece, renamed_file)))
            os.remove((os.path.join(rootdir_piece, wav_file)))  # Delete original
file_to_store = open(
    os.path.join(PATH_PICKLE_FOLDER, "Dic_pieces.pkl"), "wb"
)  # Saves the names of the pieces


pickle.dump(dic_title_wav, file_to_store)

file_to_store.close()
