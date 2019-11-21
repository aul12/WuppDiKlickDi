import load
import model
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data.dataloader

epochs = 100000
batch_size = 1
depth = 4

def main():
    device = torch.device("cpu")

    encoder = model.Encoder(depth)
    decoder = model.Decoder(depth)
    net = torch.nn.Sequential(encoder, decoder).to(device)
    optimizer = optim.Adadelta(net.parameters(), lr=0.01)

    dataset = load.WavDataSet(["1.wav", "2.wav"], model.downsample_factor**depth)
    dataloader = torch.utils.data.dataloader.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Starting training")
    for e in range(epochs):
        net.train()
        for batch_idx, data in enumerate(dataloader):
            optimizer.zero_grad()
            output = net(data)
            loss = F.mse_loss(output, data)
            loss.backward()
            optimizer.step()
            print(
                f'Train Epoch: {e} [{batch_idx * len(data)}/{len(dataloader.dataset)} ({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}')


if __name__ == "__main__":
    main()
