import torch
import torch.nn as nn
from .layers import unet_conv2, unet_up
from .utils import init_weights, count_param


class UNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=2, feature_scale=2, is_deconv=True, is_batchnorm=True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv1 = unet_conv2(self.in_channels, filters[0], self.is_batchnorm)
        self.conv2 = unet_conv2(filters[0], filters[1], self.is_batchnorm)
        self.conv3 = unet_conv2(filters[1], filters[2], self.is_batchnorm)
        self.conv4 = unet_conv2(filters[2], filters[3], self.is_batchnorm)
        self.center = unet_conv2(filters[3], filters[4], self.is_batchnorm)
        # upsampling
        self.up_concat4 = unet_up(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unet_up(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unet_up(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unet_up(filters[1], filters[0], self.is_deconv)
        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

        # Sigmoid
        self.sigmoid = nn.Sigmoid()

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)  # 16*512*512
        maxpool1 = self.maxpool(conv1)  # 16*256*256

        conv2 = self.conv2(maxpool1)  # 32*256*256
        maxpool2 = self.maxpool(conv2)  # 32*128*128

        conv3 = self.conv3(maxpool2)  # 64*128*128
        maxpool3 = self.maxpool(conv3)  # 64*64*64

        conv4 = self.conv4(maxpool3)  # 128*64*64
        maxpool4 = self.maxpool(conv4)  # 128*32*32

        center = self.center(maxpool4)  # 256*32*32
        up4 = self.up_concat4(center, conv4)  # 128*64*64
        up3 = self.up_concat3(up4, conv3)  # 64*128*128
        up2 = self.up_concat2(up3, conv2)  # 32*256*256
        up1 = self.up_concat1(up2, conv1)  # 16*512*512

        return self.final(up1)


class UNet_Nested(nn.Module):
    def __init__(self, in_channels=3, n_classes=2, feature_scale=2, is_deconv=True, is_batchnorm=True, is_ds=True):
        super(UNet_Nested, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.is_ds = is_ds

        # filters = [64, 128, 256, 512, 1024]
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv00 = unet_conv2(self.in_channels, filters[0], self.is_batchnorm)
        self.conv10 = unet_conv2(filters[0], filters[1], self.is_batchnorm)
        self.conv20 = unet_conv2(filters[1], filters[2], self.is_batchnorm)
        self.conv30 = unet_conv2(filters[2], filters[3], self.is_batchnorm)
        self.conv40 = unet_conv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat01 = unet_up(filters[1], filters[0], self.is_deconv)
        self.up_concat11 = unet_up(filters[2], filters[1], self.is_deconv)
        self.up_concat21 = unet_up(filters[3], filters[2], self.is_deconv)
        self.up_concat31 = unet_up(filters[4], filters[3], self.is_deconv)

        self.up_concat02 = unet_up(filters[1], filters[0], self.is_deconv, 3)
        self.up_concat12 = unet_up(filters[2], filters[1], self.is_deconv, 3)
        self.up_concat22 = unet_up(filters[3], filters[2], self.is_deconv, 3)

        self.up_concat03 = unet_up(filters[1], filters[0], self.is_deconv, 4)
        self.up_concat13 = unet_up(filters[2], filters[1], self.is_deconv, 4)

        self.up_concat04 = unet_up(filters[1], filters[0], self.is_deconv, 5)

        # final conv (without any concat)
        self.final_1 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_2 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_3 = nn.Conv2d(filters[0], n_classes, 1)
        self.final_4 = nn.Conv2d(filters[0], n_classes, 1)

        # Sigmoid
        self.sigmoid = nn.Sigmoid()

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # column : 0
        X_00 = self.conv00(inputs)  # 16*512*512
        maxpool0 = self.maxpool(X_00)  # 16*256*256
        X_10 = self.conv10(maxpool0)  # 32*256*256
        maxpool1 = self.maxpool(X_10)  # 32*128*128
        X_20 = self.conv20(maxpool1)  # 64*128*128
        maxpool2 = self.maxpool(X_20)  # 64*64*64
        X_30 = self.conv30(maxpool2)  # 128*64*64
        maxpool3 = self.maxpool(X_30)  # 128*32*32
        X_40 = self.conv40(maxpool3)  # 256*32*32
        # column : 1
        X_01 = self.up_concat01(X_10, X_00)
        X_11 = self.up_concat11(X_20, X_10)
        X_21 = self.up_concat21(X_30, X_20)
        X_31 = self.up_concat31(X_40, X_30)
        # column : 2
        X_02 = self.up_concat02(X_11, X_00, X_01)
        X_12 = self.up_concat12(X_21, X_10, X_11)
        X_22 = self.up_concat22(X_31, X_20, X_21)
        # column : 3
        X_03 = self.up_concat03(X_12, X_00, X_01, X_02)
        X_13 = self.up_concat13(X_22, X_10, X_11, X_12)
        # column : 4
        X_04 = self.up_concat04(X_13, X_00, X_01, X_02, X_03)

        # final layer
        final_1 = self.final_1(X_01)
        final_2 = self.final_2(X_02)
        final_3 = self.final_3(X_03)
        final_4 = self.final_4(X_04)

        final = (final_1 + final_2 + final_3 + final_4) / 4

        if self.is_ds:
            return final
        else:
            return final_4


def test_u_net():
    print('#### Test Case ###')
    from torch.autograd import Variable

    x = Variable(torch.rand(1, 3, 512, 512)).cuda()
    model = UNet().cuda()
    param = count_param(model)
    y = model(x)
    print('Output shape:', y.shape)
    print(y)
    print('UNet totoal parameters: %.2fM (%d)' % (param / 1e6, param))


def test_u_net_nested():
    print('#### Test Case ###')
    from torch.autograd import Variable

    x = Variable(torch.rand(1, 3, 512, 512)).cuda()
    model = UNet_Nested().cuda()
    param = count_param(model)
    y = model(x)
    print('Output shape:', y.shape)
    print(y)
    print('UNet++ totoal parameters: %.2fM (%d)' % (param / 1e6, param))

