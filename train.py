import os
import torch
import torch.optim as optim
import torch.nn as nn
from model.utils import load_dataloader
from model.u_net import UNet_Nested
from model.loss import GeneralizedSoftDiceLoss
from matplotlib import pyplot as plt


def main():
    training = load_dataloader('./datasets/datasets.txt', batch_size=256)
    net = UNet_Nested()
    if torch.cuda.is_available():
        net.cuda()
    torch.manual_seed(202221090109)
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    if os.path.exists('./data-21.pth'):
        model_, optimizer_ = torch.load('./data-21.pth')
        net.load_state_dict(model_)
        optimizer.load_state_dict(optimizer_)
    criterion = nn.CrossEntropyLoss()
    dice = GeneralizedSoftDiceLoss()

    running_loss = 0.0
    for epoch in range(40):
        for i, data in enumerate(training, 0):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels) + dice(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 50 == 49:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss}')
                running_loss = 0.0
                img_1 = outputs[0].permute(1, 2, 0)
                img_1 = img_1.cpu().detach().numpy()
                plt.imshow(img_1[:, :, 0])
                plt.title('Pre 0')
                plt.show()
                plt.imshow(img_1[:, :, 1])
                plt.title('Pre 1')
                plt.show()
                plt.imshow(labels[0].cpu().detach().numpy())
                plt.title('Ground Truth')
                plt.show()
            print(f'single: {loss.item()}')

    torch.save((
        net.state_dict(),
        optimizer.state_dict()
    ), './data-21.pth')
    print('Finished')


if __name__ == '__main__':
    main()
