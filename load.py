import torch.utils.data.dataset
import torch
import scipy.io.wavfile
import numpy as np
import glob

sample_rate = 20000

class WavDataSet(torch.utils.data.dataset.Dataset):
    def __init__(self, path, factor, device):
        self.fileNames = list(glob.glob(path+"/*.wav"))
        self.factor = factor
        self.files = dict()
        self.device = device
        samples = sample_rate * 30
        self.end = samples - (samples % self.factor)
        print("Actual time: %f (%d samples)" % (self.end/sample_rate, self.end))

    def read_file(self, fname):
        rate, data = scipy.io.wavfile.read(fname)

        if rate != sample_rate:
            raise ValueError("Sample rate is %d not %d for file %s" % (rate, sample_rate, self.fileNames[idx]))

        data = data[0:self.end]
        data_min = np.min(data)
        data_max = np.max(data)
        data = list(map(lambda x: -x / data_min if x < 0 else x / data_max, data))

        noise = np.random.randn(len(data)) * 0.1

        data_noise = data + noise

        data = np.expand_dims(data, axis=0)
        data = torch.Tensor(data).to(self.device)

        data_noise = np.expand_dims(data_noise, axis=0)
        data_noise = torch.Tensor(data).to(self.device)

        return (data_noise, data)

    def __len__(self):
        return len(self.fileNames)

    def __getitem__(self, idx):
        fname = self.fileNames[idx]
        if fname in self.files:
            return self.files[fname]
        else:
            data = self.read_file(fname)
            self.files[fname] = data
            return data

