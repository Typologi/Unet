import torch
import torch.optim as optim
from torch.nn.functional import cross_entropy
from model.utils import load_dataloader
from model.u_net import UNet_Nested
from matplotlib import pyplot as plt


def main():
    training = load_dataloader('./datasets/datasets.txt')
    test = load_dataloader('./datasets/testsets.txt')
    net = UNet_Nested()
    if torch.cuda.is_available():
        net.cuda()
    torch.manual_seed(202221090109)
    optimizer = optim.Adam(net.parameters(), lr=0.0005)

    running_loss = 0.0
    for epoch in range(20):
        for i, data in enumerate(training, 0):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs = net(inputs)
            print(outputs.shape)
            print(labels.shape)
            loss = cross_entropy(outputs.squeeze(1), labels.squeeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 3 == 2:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss}')
                running_loss = 0.0
                plt.imshow(outputs[0].permute(1, 2, 0).cpu().detach().numpy())
                plt.show()
                plt.imshow(labels[0].permute(1, 2, 0).cpu().detach().numpy())
                plt.show()
            print(f'loss: {loss.item()}')

    print('Finished')


if __name__ == '__main__':
    main()
