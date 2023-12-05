import os
import nibabel as nib
from torch.utils.data import Dataset

class BraTSDataset(Dataset):
    def __init__(self, data_dir, sequences=['t1ce'], transform=None):
        self.data_dir = data_dir
        self.sequences = sequences
        self.transform = transform
        self.file_list = self._create_file_list()

    def _create_file_list(self):
        file_list = []
        for folder_name in os.listdir(self.data_dir):
            folder_path = os.path.join(self.data_dir, folder_name)
            if os.path.isdir(folder_path):
                images = {}
                for seq in self.sequences:
                    for file_name in os.listdir(folder_path):
                        if seq in file_name and file_name.endswith('.nii'):
                            images[seq] = os.path.join(folder_path, file_name)
                if images:  # if the dictionary is not empty
                    file_list.append(images)
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        images = self.file_list[idx]
        data = {}
        for seq, file_path in images.items():
            image = nib.load(file_path).get_fdata()
            if self.transform:
                image = self.transform(image)
            data[seq] = image
        return data
