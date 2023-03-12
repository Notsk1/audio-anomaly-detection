"""
Module contains function for preprocessing of the data before model is run
"""
import pickle
import shutil
import pathlib
import random
import librosa

import numpy as np

from librosa.core import load as lb_load, stft
from librosa.filters import mel

SR = 16000

def split_data():
    """
    Create train, test and validation split for data
    """
    relative_path_to_root = 'data/raw/fan'
    root_object_path = pathlib.Path(pathlib.Path().absolute(), relative_path_to_root)
    for id_folder in list(root_object_path.iterdir()):
        # Get abnormal and normal folders
        abnormal_folder = pathlib.Path(pathlib.Path(id_folder, 'abnormal'))
        normal_folder = pathlib.Path(pathlib.Path(id_folder, 'normal'))

        # Add train test and val dirs to folder
        pathlib.Path(f"{abnormal_folder.absolute()}/testing").mkdir(parents=True, exist_ok=True)
        pathlib.Path(f"{abnormal_folder.absolute()}/validation").mkdir(parents=True, exist_ok=True)

        pathlib.Path(f"{normal_folder.absolute()}/training").mkdir(parents=True, exist_ok=True)
        pathlib.Path(f"{normal_folder.absolute()}/testing").mkdir(parents=True, exist_ok=True)
        pathlib.Path(f"{normal_folder.absolute()}/validation").mkdir(parents=True, exist_ok=True)


        abnormal_files_list = list(abnormal_folder.iterdir())
        normal_files_list = list(normal_folder.iterdir())

        abnormal_file_numbers = list(range(0, len(abnormal_files_list)))
        normal_files_numbers = list(range(0, len(normal_files_list)))
        random.shuffle(abnormal_file_numbers)
        random.shuffle(normal_files_numbers)

        abnormal_test_split = abnormal_file_numbers[:int(len(abnormal_files_list)*0.6)]
        abnormal_validation_split = abnormal_file_numbers[int(len(abnormal_files_list)*0.6):]
        

        normal_training_split = normal_files_numbers[:int(len(normal_files_list)*0.65)]
        normal_test_split = normal_files_numbers[int(len(normal_files_list)*0.65):int(len(normal_files_list)*0.85)]
        normal_validation_split = normal_files_numbers[int(len(normal_files_list)*0.85):]

        abnormal_test_split = [str(item).zfill(8) for item in abnormal_test_split]
        abnormal_validation_split = [str(item).zfill(8) for item in abnormal_validation_split]

        normal_training_split = [str(item).zfill(8) for item in normal_training_split]
        normal_test_split = [str(item).zfill(8) for item in normal_test_split]
        normal_validation_split = [str(item).zfill(8) for item in normal_validation_split]

        for file in abnormal_files_list:
            if file.stem in abnormal_test_split:
                shutil.copyfile(file.absolute(), f"{abnormal_folder.absolute()}/testing/{file.stem}.wav")
            elif file.stem in abnormal_validation_split:
                shutil.copyfile(file.absolute(), f"{abnormal_folder.absolute()}/validation/{file.stem}.wav")

        for file in normal_files_list:
            if file.stem in normal_test_split:
                shutil.copyfile(file.absolute(), f"{normal_folder.absolute()}/testing/{file.stem}.wav")
            elif file.stem in normal_validation_split:
                shutil.copyfile(file.absolute(), f"{normal_folder.absolute()}/validation/{file.stem}.wav")
            elif file.stem in normal_training_split:
                shutil.copyfile(file.absolute(), f"{normal_folder.absolute()}/training/{file.stem}.wav")
            

def serialize(mel_or_big_mel):
    """
    Calculate mel spectrogram and serialize the results

    Arguments:
    - mel_or_big_mel: Define which method is used for extracting audio feature
    """
    relative_path_to_data_root = 'data/raw/fan'
    root_object_path = pathlib.Path(pathlib.Path().absolute(), relative_path_to_data_root)
    for id_folder in list(root_object_path.iterdir()):
        # Get data from abnormal folder
        abnormal_folder = pathlib.Path(pathlib.Path(id_folder, 'abnormal'))
        abnormal_folder_list = list(abnormal_folder.iterdir())
        for abnormal_sub_folder in abnormal_folder_list:
            for abnormal_file in list(abnormal_sub_folder.iterdir()):
                features_and_classes = {"features":[],
                        "class":[]}
                # Get audio data and convert to mel and then serialize
                audio_data, sr = get_audio_file_data(abnormal_file)
                if mel_or_big_mel == 'mel':
                    audio_features = extract_mel_band_energies(audio_data)
                elif mel_or_big_mel == 'big_mel':
                    pure_audio_features = extract_mel_band_energies(audio_data, n_mels = 128)
                    audio_features = librosa.power_to_db(pure_audio_features)

                features_and_classes["features"] = audio_features
                features_and_classes["class"] = 1 # Abnormal samples are class 1

                # Get new path and make parent directories if necessary
                new_path = abnormal_file.absolute().as_posix().replace('raw/fan', f'serialized/fan/{mel_or_big_mel}')
                pathlib.Path(new_path).parents[0].mkdir(parents=True, exist_ok=True)
                file = open(new_path.replace(f'{abnormal_file.stem}.wav', f'{abnormal_file.stem}_{mel_or_big_mel}.pickle'), 'wb')
                pickle.dump(features_and_classes, file)
                file.close()

        # Get data from normal folder
        normal_folder = pathlib.Path(pathlib.Path(id_folder, 'normal'))
        normal_files_list = list(normal_folder.iterdir())

        normal_means = []
        normal_stds = []
        db_normal_means = []
        db_normal_stds = []
        for normal_sub_folder in normal_files_list:
            for normal_file in list(normal_sub_folder.iterdir()):
                features_and_classes = {"features":[],
                        "class":[]}
                # Get audio data and convert to mel and then serialize
                audio_data, sr = get_audio_file_data(normal_file)
                if mel_or_big_mel == 'mel':
                    audio_features = extract_mel_band_energies(audio_data)
                elif mel_or_big_mel == 'big_mel':
                    pure_audio_features = extract_mel_band_energies(audio_data, n_mels = 128)
                    audio_features = librosa.power_to_db(pure_audio_features)

                features_and_classes["features"] = audio_features
                features_and_classes["class"] = 0 # Normal samples are class 0

                # Calculate the mean and std for normalization
                normal_means.append(np.mean(pure_audio_features))
                normal_stds.append(np.std(pure_audio_features))

                db_normal_means.append(np.mean(audio_features))
                db_normal_stds.append(np.std(audio_features))

                # Get new path and make parent directories if necessary
                new_path = normal_file.absolute().as_posix().replace('raw/fan', f'serialized/fan/{mel_or_big_mel}')
                pathlib.Path(new_path).parents[0].mkdir(parents=True, exist_ok=True)
                file = open(new_path.replace(f'{normal_file.stem}.wav', f'{normal_file.stem}_{mel_or_big_mel}.pickle'), 'wb')
                pickle.dump(features_and_classes, file)
                file.close()

    print(np.mean(np.array(normal_means)))
    print(np.mean(np.array(normal_stds)))
    print(np.mean(np.array(db_normal_means)))
    print(np.mean(np.array(db_normal_stds)))

def extract_stft(audio_file, n_fft = 1024, hop_length = 512):
    """Extracts and returns the magnitude information of STFT from the 'audio_file' audio file"""
    return np.abs(stft(y=audio_file, n_fft=n_fft, hop_length=hop_length))

def get_audio_file_data(audio_file):
    """Loads and returns the audio data from the `audio_file`
    """
    return lb_load(path=audio_file, sr=SR, mono=True)

def extract_mel_band_energies(audio_file, n_fft = 1024, hop_length = 512, n_mels = 40):
    """Extracts and returns the mel-band energies from the `audio_file` audio file.
    """
    mel_specto = librosa.feature.melspectrogram(y = audio_file, sr=SR, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2)

    return mel_specto

def main():
    # split_data()
    serialize('big_mel')
    # serialize 

if __name__ == "__main__":
    main()