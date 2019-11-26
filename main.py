import load
import model
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data.dataloader

epochs = 100000
batch_size = 1
depth = 8

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_device = torch.device("cpu")
    print("Device type: %s" % device.type)

    encoder = model.Encoder(depth)
    decoder = model.Decoder(depth)
    net = torch.nn.Sequential(encoder, decoder).to(device)
    optimizer = optim.Adadelta(net.parameters(), lr=0.01)

    dataset = load.WavDataSet("data/wav/", model.downsample_factor**depth, data_device)
    dataloader = torch.utils.data.dataloader.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Starting training")
    for e in range(epochs):
        net.train()
        loss_sum = 0
        for batch_idx, (data_noise, data) in enumerate(dataloader):
            data = data.to(device)
            data_noise = data_noise.to(device)
            optimizer.zero_grad()
            output = net(data_noise)
            loss = F.mse_loss(output, data)
            loss.backward()
            optimizer.step()
            loss_sum += loss
        print("Epoch: %d\tLoss: %f" % (e, loss_sum))
        if e % 50 == 0:
            torch.save(net.state_dict(), "checkpoint/model_%d.pth" % e)


if __name__ == "__main__":
    main()
