import torch
import torch.nn as nn
from nets.vgg import VGG16


class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        # Tensor shape is [batch_size, channels, w, h]
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)  # input2双线性上采用后通道上进行拼接
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs


class Unet(nn.Module):
    def __init__(self, num_classes=21, in_channels=3, pretrained=False):
        super(Unet, self).__init__()
        self.vgg = VGG16(pretrained=pretrained,in_channels=in_channels)  # 这里的vgg删除了平均池化和分类层(全连接层)
        in_filters = [192, 384, 768, 1024]
        out_filters = [64, 128, 256, 512]
        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        # final conv (without any concat)
        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

    def forward(self, inputs):
        feat1 = self.vgg.features[  :4 ](inputs)  # 特征层1 con1_3 输出通道64
        feat2 = self.vgg.features[4 :9 ](feat1)  # 特征层2 从Maxpool开始 输出con2_3 128通道
        feat3 = self.vgg.features[9 :16](feat2)  # 特征层3 从Maxpool开始 初始con3_3 256通道
        feat4 = self.vgg.features[16:23](feat3)  # 特征层4 从Maxpool开始 输出con4_3 512通道
        feat5 = self.vgg.features[23:-1](feat4)  # 特征层5 从Maxpool开始 输出con5_3 512通道

        up4 = self.up_concat4(feat4, feat5)  # 输出512通道，w,h与feat4一样
        up3 = self.up_concat3(feat3, up4)  # 输出256通道，w,h与feat3一样
        up2 = self.up_concat2(feat2, up3)  # 输出128通道，w,h与feat2一样
        up1 = self.up_concat1(feat1, up2)  # 输出64通道，w,h与feat1一样

        final = self.final(up1)  # 输出num_classes通道,w,h与up一样
        
        return final  # 输出和原始输入大小一样，通道数位num_classes

    def _initialize_weights(self, *stages):
        for modules in stages:
            for module in modules.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()

