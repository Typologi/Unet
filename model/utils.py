import torch
import torchvision
import torch.nn as nn
from torch.nn import init
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


# initialize the module
def init_weights(net, init_type='normal'):
    # print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


# compute model params
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


def save_checkpoint(model):
    torch.save(model.state_dict(), './ckpt/main.pth')


def load_checkpoint(model):
    model.load_state_dict(torch.load, './ckpt/main.pth')
    model.eval()
    return model


class RetinaDatasets(Dataset):
    def __init__(self, config_path):
        print('Processing Image...')
        super().__init__()
        self.images = []
        self.labels = []
        resize = torchvision.transforms.Resize((512, 512))
        with open(config_path) as f:
            for line in f.readlines():
                if line == '':
                    continue
                image, label = line.replace('\n', '').replace('\r', '').split('|')
                img = resize(torchvision.io.read_image(image,
                                                       mode=torchvision.io.ImageReadMode.RGB)).float() / 255.0
                label = resize(torchvision.io.read_image(label,
                                                         mode=torchvision.io.ImageReadMode.GRAY)).float() / 255.0
                label = label.long().squeeze(0)
                for i in range(8):
                    height = 64 * i
                    for j in range(8):
                        width = 64 * j
                        self.images.append(img[:, width: width + 64, height: height + 64])
                        self.labels.append(label[width: width + 64, height: height + 64])
        print('Finished Processing...')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def load_datasets(path):
    return RetinaDatasets(path)


def load_dataloader(path, num_workers=16, batch_size=256, shuffle=True):
    return DataLoader(dataset=load_datasets(path), shuffle=shuffle,
                      batch_size=batch_size, num_workers=num_workers)


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

