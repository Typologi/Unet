import os.path

import torch.cuda

from model.utils import load_dataloader
from model.u_net import UNet_Nested
from matplotlib import pyplot as plt


def main():
    net = UNet_Nested()
    if os.path.exists('./data-21.pth'):
        model_, _ = torch.load('./data-21.pth')
        net.load_state_dict(model_)
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    print('OK')
    test = load_dataloader('./datasets/testsets.txt', batch_size=64, shuffle=False)

    result = []
    with torch.no_grad():
        for i, data in enumerate(test, 0):
            x, _ = data
            if torch.cuda.is_available():
                x = x.cuda()
            result.append(net(x))

    for index, image_array in enumerate(result, 0):
        single_image = torch.zeros((512, 512))
        for i in range(8):
            height = 64 * i
            for j in range(8):
                width = 64 * j
                single_image[width:(width + 64), height:(height + 64)] = image_array[i * 8 + j, 1, :, :]
        single_image[single_image < 0] = 0.0
        single_image[single_image > 0] = 255.0
        img_1 = single_image.cpu().numpy()
        plt.imshow(img_1)
        plt.title('Pre 1')
        plt.show()


if __name__ == '__main__':
    main()

