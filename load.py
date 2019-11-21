import torch.utils.data.dataset
import scipy.io.wavfile
import numpy as np


class WavDataSet(torch.utils.data.dataset.Dataset):
    def __init__(self, file_names):
        self.fileNames = file_names

    def __len__(self):
        return len(self.fileNames)

    def __getitem__(self, idx):
        rate, data = scipy.io.wavfile.read(self.fileNames[idx])

        if rate != 44100:
            raise ValueError("Sample rate is not 44100 for file %s" % (self.fileNames[idx]))

        if data.shape[1] != 1:
            data = data[:, 0]
        end = len(data) - (len(data) % 2**4)
        data = data[0:end]
        data_min = np.min(data)
        data_max = np.max(data)
        data = list(map(lambda x: -x / data_min if x < 0 else x / data_max, data))
        data = np.expand_dims(data, axis=0)

        return data
