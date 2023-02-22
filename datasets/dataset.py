"""
File defines a dataset class for anomaly detection model.
"""

from torch.utils.data import Dataset
import pathlib
import numpy as np
import pickle

class AnomalyDataset(Dataset):
    """
    Dataset for anomaly detection on the MIMI dataset. Can be used for mel and stft data
    """

    def __init__(self, split_type, data_folder, object_id = "", load_into_memory = True) -> None:
        """
        Initialize anomaly detection dataset

        Arguments:
        - split_type: Type of split. Either training, (validation) or testing
        - data_folder: Root folder which contains the different id folders
        - object_id: Defines the object id which is used for training. If nothing/empty string
                     given, all object ids used for training
        - load_into_memory: Define if files are load into memory before training
        """
        super().__init__()
        self.data_folder = pathlib.Path(data_folder)
        self.split_type = split_type
        self.object_id = object_id
        self.load_into_memory = load_into_memory

        # Contains list of files if load_into_memory == False, otherwise list of features and
        # classes, one per file
        self.files = []

        self.get_files()
    
    @staticmethod
    def _load_file(file_path):
        """Loads a file using pathlib.Path
        
        Arguments:
        - file_path: pathlib path object
        """
        with file_path.open('rb') as f:
            return pickle.load(f)

    def get_files(self):
        """
        Save file paths or file data from the given folder depending on the settings
        """
        for id_folder in list(self.data_folder.iterdir()):
            # Load data/files from only wanted objects
            if id_folder.stem == self.object_id or self.object_id == '':
                # Get data from abnormal folder if validation or testing
                abnormal_folder = pathlib.Path(pathlib.Path(id_folder, 'abnormal'))
                abnormal_folder_list = list(abnormal_folder.iterdir())

                for abnormal_sub_folder in abnormal_folder_list:
                    # Get files only from matching split type folders
                    if abnormal_sub_folder.stem == self.split_type:
                        for abnormal_file in list(abnormal_sub_folder.iterdir()):
                            if self.load_into_memory:
                                self.files.append(self._load_file(abnormal_file))
                            else:
                                self.files.append(abnormal_file)
                
                # Get data from normal folders for all split types
                normal_folder = pathlib.Path(pathlib.Path(id_folder, 'normal'))
                normal_folder_list = list(normal_folder.iterdir())

                for normal_sub_folder in normal_folder_list:
                    # Get files only from matching split type folders
                    if normal_sub_folder.stem == self.split_type:
                        for normal_file in list(normal_sub_folder.iterdir()):
                            if self.load_into_memory:
                                self.files.append(self._load_file(normal_file))
                            else:
                                self.files.append(normal_file)
    
    def __len__(self):
        """Return the length of the dataset"""
        return len(self.files)
    
    def __getitem__(self, item):
        """Returns an item from the dataset.

        Arguments:
        - item: Index of the item.
        """
        if self.load_into_memory:
            the_item = self.files[item]
        else:
            the_item = self._load_file(self.files[item])

        return the_item['features'], the_item['class']


