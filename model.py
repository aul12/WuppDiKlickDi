import torch.nn as nn
import torch

downsample_factor = 4
filter_size = 9
filter_number = 32

class Encoder(nn.Module):
    def __init__(self, num_layers):
        super(Encoder, self).__init__()
        self.conv = nn.ModuleList()
        self.maxpool = nn.ModuleList()

        for c in range(num_layers):
            in_channels = 1 if c == 0 else filter_number
            self.conv.append(nn.Conv1d(in_channels=in_channels, out_channels=filter_number, kernel_size=filter_size, padding=filter_size//2))
            self.maxpool.append(nn.MaxPool1d(kernel_size=downsample_factor))

        self.conv.append(nn.Conv1d(in_channels=filter_number, out_channels=filter_number, kernel_size=filter_size, padding=filter_size//2))

    def forward(self, x):
        for conv, maxpool in zip(self.conv, self.maxpool):
            x = conv(x)
            x = torch.tanh(x)
            x = maxpool(x)
        return self.conv[-1](x)


class Decoder(nn.Module):
    def __init__(self, num_layers):
        super(Decoder, self).__init__()
        self.conv = nn.ModuleList()
        self.upsample = nn.ModuleList()

        for c in range(num_layers):
            out_channels = 1 if c == num_layers - 1 else filter_number
            self.conv.append(nn.Conv1d(in_channels=filter_number, out_channels=out_channels, kernel_size=filter_size, padding=filter_size//2))
            self.upsample.append(nn.Upsample(scale_factor=downsample_factor))

    def forward(self, x):
        for num, (upsample, conv) in enumerate(zip(self.upsample, self.conv)):
            x = upsample(x)
            x = conv(x)
            x = torch.tanh(x)

        return x

