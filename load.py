import torch.utils.data.dataset
import scipy.io.wavfile
import numpy as np
import glob


class WavDataSet(torch.utils.data.dataset.Dataset):
    def __init__(self, path, factor):
        self.fileNames = list(glob.glob(path+"/*.wav"))
        self.factor = factor

    def __len__(self):
        return len(self.fileNames)

    def __getitem__(self, idx):
        rate, data = scipy.io.wavfile.read(self.fileNames[idx])

        if rate != 20000:
            raise ValueError("Sample rate is %d not 44100 for file %s" % (rate, self.fileNames[idx]))

        end = len(data) - (len(data) % self.factor)
        data = data[0:end]
        data_min = np.min(data)
        data_max = np.max(data)
        data = list(map(lambda x: -x / data_min if x < 0 else x / data_max, data))
        data = np.expand_dims(data, axis=0)

        return data
