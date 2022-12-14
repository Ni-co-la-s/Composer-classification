# Composer-classification
Classification of composers from the songs of the Maestro dataset (downloaded at https://www.kaggle.com/datasets/jackvial/themaestrodatasetv2)

# Description
The maestro dataset contains more than 200 hours of piano playing, in the same conditions. It has them in WAV and MIDI but in this case, we only considered the audio.
This repository aims to classify 20 seconds extracts of a piece as a certain composer. Only the composers with more than 20 different pieces were kept, duplicates were removed.
The maestro dataset is originally in the WAV format but in order to save space, the kaggle dataset was converted to MP3 (which may lead to quality loss, especially since the extracts are then converted to WAV again in order to be exploited with librosa). Adapting the code to fit the original data would not be particularly difficult.

# Getting Started
## Dependencies
*librosa *soundfile *sklearn *pydub

## Installing
Clone this project or place the 5 files (audio_processing.py, creation_extracts.py, edit_csv.py, grid_search_svm.py and utils.txt) in the same folder. Download the kaggle dataset and put it in the same folder as the files.
In utils.py, change if needed the root_dir elements, the duration of the extracts and the other parameters.

## Executing program
In order, use:
-edit_csv.py to filter out the duplicates and choose the composers to keep in the csv
-creation_extracts.py to create the training and validation wav files
-audio_processing.py to create the pickle files with the mfcc/chroma vectors and perform data augmentation
-grid_search_svm.py to find the optimal parameters for the SVM

# Results

To evaluate the results of the model, the pieces were split in a 80/20 training-testing/validation. So the extracts used in the validation were not an influence during the training.
For 4 composers (Liszt, Mozart, Brahms and Bach), we obtain a 60% accuracy, which is a lot better than the 25% we would obtain randomly.
|![classification_composer](https://user-images.githubusercontent.com/96898279/198711677-0ad44e7f-1f62-475e-86ce-c40f5688e162.png)
|:--:| 
| *Figure 1: Confusion matrix for the SVM model* |




