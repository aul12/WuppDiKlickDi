import load
import model
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data.dataloader
import scipy.io.wavfile

batch_size = 1
depth = 8

def main():
    device = torch.device("cpu")
    print("Device type: %s" % device.type)

    encoder = model.Encoder(depth)
    decoder = model.Decoder(depth)
    net = torch.nn.Sequential(encoder, decoder).to(device)
    net.load_state_dict(torch.load("checkpoint/model_3150.pth"))
    net.eval()

    dataset = load.WavDataSet("data/wav/", model.downsample_factor**depth, device)
    dataloader = torch.utils.data.dataloader.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for batch_idx, data in enumerate(dataloader):
        data = data.to(device)
        output = net(data)
        scipy.io.wavfile.write("out/%d.wav" % batch_idx, load.sample_rate, output.data.numpy())
        print("Finished %d" % batch_idx)


if __name__ == "__main__":
    main()
