import torch
from torch import nn
import torch.nn.functional as F
from tool.torch_utils import *
from tool.yolo_layer import YoloLayer


class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, x, target_size, inference=False):
        assert (x.data.dim() == 4)
        # _, _, tH, tW = target_size

        if inference:

            #B = x.data.size(0)
            #C = x.data.size(1)
            #H = x.data.size(2)
            #W = x.data.size(3)

            return x.view(x.size(0), x.size(1), x.size(2), 1, x.size(3), 1).\
                    expand(x.size(0), x.size(1), x.size(2), target_size[2] // x.size(2), x.size(3), target_size[3] // x.size(3)).\
                    contiguous().view(x.size(0), x.size(1), target_size[2], target_size[3])
        else:
            return F.interpolate(x, size=(target_size[2], target_size[3]), mode='nearest')


class Conv_Bn_Activation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, bn=True, bias=False):
        super().__init__()
        pad = (kernel_size - 1) // 2

        self.conv = nn.ModuleList()
        if bias:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad))
        else:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False))
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))
        if activation == "mish":
            self.conv.append(Mish())
        elif activation == "relu":
            self.conv.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "linear":
            pass
        else:
            print("activate error !!! {} {} {}".format(sys._getframe().f_code.co_filename,
                                                       sys._getframe().f_code.co_name, sys._getframe().f_lineno))

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x


class ResBlock(nn.Module):
    """
    Sequential residual blocks each of which consists of \
    two convolution layers.
    Args:
        ch (int): number of input and output channels.
        nblocks (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """

    def __init__(self, ch, nblocks=1, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(Conv_Bn_Activation(ch, ch, 1, 1, 'mish'))
            resblock_one.append(Conv_Bn_Activation(ch, ch, 3, 1, 'mish'))
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x


class DownSample1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(3, 32, 3, 1, 'mish')

        self.conv2 = Conv_Bn_Activation(32, 64, 3, 2, 'mish')
        self.conv3 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        # [route]
        # layers = -2
        self.conv4 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')

        self.conv5 = Conv_Bn_Activation(64, 32, 1, 1, 'mish')
        self.conv6 = Conv_Bn_Activation(32, 64, 3, 1, 'mish')
        # [shortcut]
        # from=-3
        # activation = linear

        self.conv7 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        # [route]
        # layers = -1, -7
        self.conv8 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # route -2
        x4 = self.conv4(x2)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        # shortcut -3
        x6 = x6 + x4

        x7 = self.conv7(x6)
        # [route]
        # layers = -1, -7
        x7 = torch.cat([x7, x3], dim=1)
        x8 = self.conv8(x7)
        return x8


class DownSample2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(64, 128, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')
        # r -2
        self.conv3 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')

        self.resblock = ResBlock(ch=64, nblocks=2)

        # s -3
        self.conv4 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        # r -1 -10
        self.conv5 = Conv_Bn_Activation(128, 128, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(128, 256, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(256, 128, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(256, 128, 1, 1, 'mish')

        self.resblock = ResBlock(ch=128, nblocks=8)
        self.conv4 = Conv_Bn_Activation(128, 128, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(256, 256, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(256, 512, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(512, 256, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(512, 256, 1, 1, 'mish')

        self.resblock = ResBlock(ch=256, nblocks=8)
        self.conv4 = Conv_Bn_Activation(256, 256, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(512, 512, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(512, 1024, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(1024, 512, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(1024, 512, 1, 1, 'mish')

        self.resblock = ResBlock(ch=512, nblocks=4)
        self.conv4 = Conv_Bn_Activation(512, 512, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(1024, 1024, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class Neck(nn.Module):
    def __init__(self, inference=False):
        super().__init__()
        self.inference = inference

        self.conv1 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv2 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv3 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        # SPP
        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)

        # R -1 -3 -5 -6
        # SPP
        self.conv4 = Conv_Bn_Activation(2048, 512, 1, 1, 'leaky')
        self.conv5 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv6 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv7 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        # UP
        self.upsample1 = Upsample()
        # R 85
        self.conv8 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        # R -1 -3
        self.conv9 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv10 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv11 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv12 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv13 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv14 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        # UP
        self.upsample2 = Upsample()
        # R 54
        self.conv15 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        # R -1 -3
        self.conv16 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.conv17 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv18 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.conv19 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv20 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')

    def forward(self, input, downsample4, downsample3, inference=False):

        # ! input       = (B, 1024, 19, 19)
        # ! downsample4 = (B, 512, 38, 38)
        # ! downsample3 = (B, 256, 76, 76)

        # ! d5 features에 대한 refine 수행
        # ! 1x1, 3x3(w/ padding), 1x1 convolution 적용하고 채널 수 감소시킴
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # ! x3 = (B, 512, 19, 19)

        # SPP
        # ! SPP를 적용하여 refined x3로부터 고정된 크기의 features 추출
        m1 = self.maxpool1(x3)
        m2 = self.maxpool2(x3)
        m3 = self.maxpool3(x3)
        # ! m1 = (B, 512, 19, 19) -> kernel size = 5
        # ! m2 = (B, 512, 19, 19) -> kernel size = 9
        # ! m3 = (B, 512, 19, 19) -> kernel size = 13

        # ! SPP 적용 결과 concat
        spp = torch.cat([m3, m2, m1, x3], dim=1)
        # ! spp = (B, 2048, 19, 19)

        # SPP end
        # ! SPP 적용 완료 후 추가적인 convolution 진행
        # ! 1x1, 3x3(w/ padding), 1x1 convolution 적용하고 채널 수 감소시킴
        x4 = self.conv4(spp)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        # ! x6 = (B, 512, 19, 19)

        # ! top-down 전달을 위해 1x1 convolution으로 채널 수 감소
        x7 = self.conv7(x6)
        # ! x7 = (B, 256, 19, 19)

        # UP
        # ! top-down 전달을 위해 upsampling
        up = self.upsample1(x7, downsample4.size(), self.inference)
        # ! up = (B, 256, 38, 38)

        # R 85
        # ! d4 features에 대한 refine 수행
        x8 = self.conv8(downsample4)
        # ! x8 = (B, 256, 38, 38)
        # R -1 -3

        # ! Upsamling된 top-down features와 refine된 d4 features 결합
        x8 = torch.cat([x8, up], dim=1)
        # ! x8 = (B, 512, 38, 38)

        # ! 결합 완료 후 추가적인 convolution 진행
        # ! 1x1, 3x3(w/ padding), 1x1, 3x3(w/ padding), 1x1 convolution 적용하고 채널 수 감소시킴
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        # ! x13 = (B, 256, 38, 38)

        # ! top-down 전달을 위해 1x1 convolution으로 채널 수 감소
        x14 = self.conv14(x13)
        # ! x14 = (B, 128, 38, 38)

        # UP
        # ! top-down 전달을 위해 upsampling
        up = self.upsample2(x14, downsample3.size(), self.inference)
        # ! up = (B, 128, 76, 76)

        # R 54
        # ! d3 features에 대한 refine 수행
        x15 = self.conv15(downsample3)
        # ! x15 = (B, 128, 76, 76)

        # R -1 -3
        # ! Upsamling된 top-down features와 refine된 d3 features 결합
        x15 = torch.cat([x15, up], dim=1)
        # ! x15 = (B, 256, 76, 76)

        # ! 결합 완료 후 추가적인 convolution 진행
        # ! 1x1, 3x3(w/ padding), 1x1, 3x3(w/ padding), 1x1 convolution 적용하고 채널 수 감소시킴
        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        x19 = self.conv19(x18)
        x20 = self.conv20(x19)
        # ! x20 = (B, 128, 76, 76)

        # ! 3가지 scale의 refined feature maps를 출력함

        # ! x20 = (B, 128, 76, 76)
        # ! x13 = (B, 256, 38, 38)
        # ! x6  = (B, 512, 19, 19)

        return x20, x13, x6


class Yolov4Head(nn.Module):
    def __init__(self, output_ch, n_classes, inference=False):
        super().__init__()
        self.inference = inference

        self.conv1 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv2 = Conv_Bn_Activation(256, output_ch, 1, 1, 'linear', bn=False, bias=True)

        self.yolo1 = YoloLayer(
                                anchor_mask=[0, 1, 2], num_classes=n_classes,
                                anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
                                num_anchors=9, stride=8)

        # R -4
        self.conv3 = Conv_Bn_Activation(128, 256, 3, 2, 'leaky')

        # R -1 -16
        self.conv4 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv5 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv6 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv7 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv8 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv9 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv10 = Conv_Bn_Activation(512, output_ch, 1, 1, 'linear', bn=False, bias=True)
        
        self.yolo2 = YoloLayer(
                                anchor_mask=[3, 4, 5], num_classes=n_classes,
                                anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
                                num_anchors=9, stride=16)

        # R -4
        self.conv11 = Conv_Bn_Activation(256, 512, 3, 2, 'leaky')

        # R -1 -37
        self.conv12 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv13 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv14 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv15 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv16 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv17 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv18 = Conv_Bn_Activation(1024, output_ch, 1, 1, 'linear', bn=False, bias=True)
        
        self.yolo3 = YoloLayer(
                                anchor_mask=[6, 7, 8], num_classes=n_classes,
                                anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
                                num_anchors=9, stride=32)

    def forward(self, input1, input2, input3):

        # ! input1 = (B, 128, 76, 76)
        # ! input2 = (B, 256, 38, 38)
        # ! input3 = (B, 512, 19, 19)

        # ! high-resolution feature maps인 input1에서 예측 수행
        # ! PASCAL VOC를 사용한다고 가정하고, n_classes = 20임
        # ! 따라서, 예측값은 y = (B, 75, S, S)이고,
        # ! 75 = (bbox 좌표 수 + confidence score + class 수) * feature maps 수 = (4 + 1 + 20) * 3
        x1 = self.conv1(input1)
        x2 = self.conv2(x1)
        # ! x2 = (B, 75, 76, 76)

        # ! bottom-up pathway 적용을 위해 convolution 수행 (downsampling)
        x3 = self.conv3(input1)
        # ! x3 = (B, 256, 38, 38)

        # R -1 -16
        # ! Downsampling된 bottom-up features와 mid-resolution feature maps 결합
        x3 = torch.cat([x3, input2], dim=1)
        # ! x3 = (B, 512, 38, 38)

        # ! 결합 완료 후 추가적인 convolution 진행
        # ! 1x1, 3x3(w/ padding), 1x1, 3x3(w/ padding), 1x1 convolution 적용
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        # ! x8 = (B, 256, 38, 38)

        # ! mid-resolution feature maps에서 예측 수행
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        # ! x10 = (B, 75, 38, 38)

        # R -4
        # ! bottom-up pathway 적용을 위해 convolution 수행 (downsampling)
        x11 = self.conv11(x8)
        # ! x11 = (B, 512, 19, 19)

        # R -1 -37
        # ! Downsampling된 bottom-up features와 low-resolution feature maps 결합
        x11 = torch.cat([x11, input3], dim=1)
        # ! x11 = (B, 1024, 19, 19)

        # ! 결합 완료 후 추가적인 convolution 진행
        # ! 1x1, 3x3(w/ padding), 1x1, 3x3(w/ padding), 1x1, 3x3(w/ padding), 1x1 convolution 적용
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)
        x15 = self.conv15(x14)
        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        # ! x17 = (B, 1024, 19, 19)

        # ! low-resolution feature maps에서 예측 수행
        x18 = self.conv18(x17)
        # ! x18 = (B, 75, 19, 19)
        
        if self.inference:
            y1 = self.yolo1(x2)
            y2 = self.yolo2(x10)
            y3 = self.yolo3(x18)

            return get_region_boxes([y1, y2, y3])
        
        else:
            
            # ! x2  = (B, 75, 76, 76)
            # ! x10 = (B, 75, 38, 38)
            # ! x18 = (B, 75, 19, 19)

            return [x2, x10, x18]


class Yolov4(nn.Module):
    def __init__(self, yolov4conv137weight=None, n_classes=80, inference=False):
        super().__init__()

        # ! 예측 채널 수
        # ! 3개의 서로 다른 scale의 feature maps마다 (x, y, w, h, confidence, class1,...,classN)를 예측함
        output_ch = (4 + 1 + n_classes) * 3

        # ! YOLO-v4는 backbone, neck, head로 구성됨
        # ! Backbone: CSPDarknet53
        # ! Neck: SPP, PAN
        # ! Head: YOLO-v3 head

        # backbone
        # ! Backbone: CSPDarknet53
        # ! 아래의 model summary 확인
        self.down1 = DownSample1()
        self.down2 = DownSample2()
        self.down3 = DownSample3()
        self.down4 = DownSample4()
        self.down5 = DownSample5()

        # neck
        # ! Neck: SPP, PAN (top-down path)
        # ! 아래의 model summary 확인
        self.neck = Neck(inference)

        # yolov4conv137
        if yolov4conv137weight:
            _model = nn.Sequential(self.down1, self.down2, self.down3, self.down4, self.down5, self.neck)
            pretrained_dict = torch.load(yolov4conv137weight)

            model_dict = _model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k1: v for (k, v), k1 in zip(pretrained_dict.items(), model_dict)}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            _model.load_state_dict(model_dict)
        
        # head
        # ! Head: YOLO-v3 head, PAN (bottom-up path)
        # ! 아래의 model summary 확인
        self.head = Yolov4Head(output_ch, n_classes, inference)

    def forward(self, input):

        # ! input = (B, 3, 608, 608)

        # ! Backbone features 추출
        d1 = self.down1(input)
        # ! d1 = (B, 64, 304, 304)

        d2 = self.down2(d1)
        # ! d2 = (B, 128, 152, 152)

        d3 = self.down3(d2)
        # ! d3 = (B, 256, 76, 76)

        d4 = self.down4(d3)
        # ! d4 = (B, 512, 38, 38)

        d5 = self.down5(d4)
        # ! d5 = (B, 1024, 19, 19)

        # ! d5, d4, d3를 multi-scale features로 사용하여 neck에 입력함
        # ! d5 = (B, 1024, 19, 19)
        # ! d4 = (B, 512, 38, 38)
        # ! d3 = (B, 256, 76, 76)
        x20, x13, x6 = self.neck(d5, d4, d3)
        # ! x20 = (B, 128, 76, 76)
        # ! x13 = (B, 256, 38, 38)
        # ! x6  = (B, 512, 19, 19)
        
        # ! 3가지 scale의 refined feature maps를 detection head로 입력함
        output = self.head(x20, x13, x6)
        # ! output -> 3가지 scale의 refined feature maps에서의 예측 tensors
        # ! output[0] = (B, 75, 76, 76)
        # ! output[1] = (B, 75, 38, 38)
        # ! output[2] = (B, 75, 19, 19)

        return output


if __name__ == "__main__":
    import sys
    import cv2

    namesfile = None
    if len(sys.argv) == 6:
        n_classes = int(sys.argv[1])
        weightfile = sys.argv[2]
        imgfile = sys.argv[3]
        height = int(sys.argv[4])
        width = int(sys.argv[5])
    elif len(sys.argv) == 7:
        n_classes = int(sys.argv[1])
        weightfile = sys.argv[2]
        imgfile = sys.argv[3]
        height = int(sys.argv[4])
        width = int(sys.argv[5])
        namesfile = sys.argv[6]
    else:
        print('Usage: ')
        print('  python models.py num_classes weightfile imgfile namefile')

    model = Yolov4(yolov4conv137weight=None, n_classes=n_classes, inference=True)

    pretrained_dict = torch.load(weightfile, map_location=torch.device('cuda'))
    model.load_state_dict(pretrained_dict)

    use_cuda = True
    if use_cuda:
        model.cuda()

    img = cv2.imread(imgfile)

    # Inference input size is 416*416 does not mean training size is the same
    # Training size could be 608*608 or even other sizes
    # Optional inference sizes:
    #   Hight in {320, 416, 512, 608, ... 320 + 96 * n}
    #   Width in {320, 416, 512, 608, ... 320 + 96 * m}
    sized = cv2.resize(img, (width, height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    from tool.utils import load_class_names, plot_boxes_cv2
    from tool.torch_utils import do_detect

    for i in range(2):  # This 'for' loop is for speed check
                        # Because the first iteration is usually longer
        boxes = do_detect(model, sized, 0.4, 0.6, use_cuda)

    if namesfile == None:
        if n_classes == 20:
            namesfile = 'data/voc.names'
        elif n_classes == 80:
            namesfile = 'data/coco.names'
        else:
            print("please give namefile")

    class_names = load_class_names(namesfile)
    plot_boxes_cv2(img, boxes[0], 'predictions.jpg', class_names)

# ! YOLOv4 backbone: CSPDarknet53
# ! mish activation function이 사용된 것에 주목: 학습 안정성 강화
# ! (3, 608, 608) 입력한 결과:
# ? ----------------------------------------------------------------
# ?         Layer (type)               Output Shape         Param #
# ? ================================================================
# ?             Conv2d-1         [-1, 32, 608, 608]             864
# ?        BatchNorm2d-2         [-1, 32, 608, 608]              64
# ?               Mish-3         [-1, 32, 608, 608]               0
# ? Conv_Bn_Activation-4         [-1, 32, 608, 608]               0
# ?             Conv2d-5         [-1, 64, 304, 304]          18,432
# ?        BatchNorm2d-6         [-1, 64, 304, 304]             128
# ?               Mish-7         [-1, 64, 304, 304]               0
# ? Conv_Bn_Activation-8         [-1, 64, 304, 304]               0
# ?             Conv2d-9         [-1, 64, 304, 304]           4,096
# ?       BatchNorm2d-10         [-1, 64, 304, 304]             128
# ?              Mish-11         [-1, 64, 304, 304]               0
# ? Conv_Bn_Activation-12         [-1, 64, 304, 304]               0
# ?            Conv2d-13         [-1, 64, 304, 304]           4,096
# ?       BatchNorm2d-14         [-1, 64, 304, 304]             128
# ?              Mish-15         [-1, 64, 304, 304]               0
# ? Conv_Bn_Activation-16         [-1, 64, 304, 304]               0
# ?            Conv2d-17         [-1, 32, 304, 304]           2,048
# ?       BatchNorm2d-18         [-1, 32, 304, 304]              64
# ?              Mish-19         [-1, 32, 304, 304]               0
# ? Conv_Bn_Activation-20         [-1, 32, 304, 304]               0
# ?            Conv2d-21         [-1, 64, 304, 304]          18,432
# ?       BatchNorm2d-22         [-1, 64, 304, 304]             128
# ?              Mish-23         [-1, 64, 304, 304]               0
# ? Conv_Bn_Activation-24         [-1, 64, 304, 304]               0
# ?            Conv2d-25         [-1, 64, 304, 304]           4,096
# ?       BatchNorm2d-26         [-1, 64, 304, 304]             128
# ?              Mish-27         [-1, 64, 304, 304]               0
# ? Conv_Bn_Activation-28         [-1, 64, 304, 304]               0
# ?            Conv2d-29         [-1, 64, 304, 304]           8,192
# ?       BatchNorm2d-30         [-1, 64, 304, 304]             128
# ?              Mish-31         [-1, 64, 304, 304]               0
# ? Conv_Bn_Activation-32         [-1, 64, 304, 304]               0
# ?       DownSample1-33         [-1, 64, 304, 304]               0
# ?            Conv2d-34        [-1, 128, 152, 152]          73,728
# ?       BatchNorm2d-35        [-1, 128, 152, 152]             256
# ?              Mish-36        [-1, 128, 152, 152]               0
# ? Conv_Bn_Activation-37        [-1, 128, 152, 152]               0
# ?            Conv2d-38         [-1, 64, 152, 152]           8,192
# ?       BatchNorm2d-39         [-1, 64, 152, 152]             128
# ?              Mish-40         [-1, 64, 152, 152]               0
# ? Conv_Bn_Activation-41         [-1, 64, 152, 152]               0
# ?            Conv2d-42         [-1, 64, 152, 152]           8,192
# ?       BatchNorm2d-43         [-1, 64, 152, 152]             128
# ?              Mish-44         [-1, 64, 152, 152]               0
# ? Conv_Bn_Activation-45         [-1, 64, 152, 152]               0
# ?            Conv2d-46         [-1, 64, 152, 152]           4,096
# ?       BatchNorm2d-47         [-1, 64, 152, 152]             128
# ?              Mish-48         [-1, 64, 152, 152]               0
# ? Conv_Bn_Activation-49         [-1, 64, 152, 152]               0
# ?            Conv2d-50         [-1, 64, 152, 152]          36,864
# ?       BatchNorm2d-51         [-1, 64, 152, 152]             128
# ?              Mish-52         [-1, 64, 152, 152]               0
# ? Conv_Bn_Activation-53         [-1, 64, 152, 152]               0
# ?            Conv2d-54         [-1, 64, 152, 152]           4,096
# ?       BatchNorm2d-55         [-1, 64, 152, 152]             128
# ?              Mish-56         [-1, 64, 152, 152]               0
# ? Conv_Bn_Activation-57         [-1, 64, 152, 152]               0
# ?            Conv2d-58         [-1, 64, 152, 152]          36,864
# ?       BatchNorm2d-59         [-1, 64, 152, 152]             128
# ?              Mish-60         [-1, 64, 152, 152]               0
# ? Conv_Bn_Activation-61         [-1, 64, 152, 152]               0
# ?          ResBlock-62         [-1, 64, 152, 152]               0
# ?            Conv2d-63         [-1, 64, 152, 152]           4,096
# ?       BatchNorm2d-64         [-1, 64, 152, 152]             128
# ?              Mish-65         [-1, 64, 152, 152]               0
# ? Conv_Bn_Activation-66         [-1, 64, 152, 152]               0
# ?            Conv2d-67        [-1, 128, 152, 152]          16,384
# ?       BatchNorm2d-68        [-1, 128, 152, 152]             256
# ?              Mish-69        [-1, 128, 152, 152]               0
# ? Conv_Bn_Activation-70        [-1, 128, 152, 152]               0
# ?       DownSample2-71        [-1, 128, 152, 152]               0
# ?            Conv2d-72          [-1, 256, 76, 76]         294,912
# ?       BatchNorm2d-73          [-1, 256, 76, 76]             512
# ?              Mish-74          [-1, 256, 76, 76]               0
# ? Conv_Bn_Activation-75          [-1, 256, 76, 76]               0
# ?            Conv2d-76          [-1, 128, 76, 76]          32,768
# ?       BatchNorm2d-77          [-1, 128, 76, 76]             256
# ?              Mish-78          [-1, 128, 76, 76]               0
# ? Conv_Bn_Activation-79          [-1, 128, 76, 76]               0
# ?            Conv2d-80          [-1, 128, 76, 76]          32,768
# ?       BatchNorm2d-81          [-1, 128, 76, 76]             256
# ?              Mish-82          [-1, 128, 76, 76]               0
# ? Conv_Bn_Activation-83          [-1, 128, 76, 76]               0
# ?            Conv2d-84          [-1, 128, 76, 76]          16,384
# ?       BatchNorm2d-85          [-1, 128, 76, 76]             256
# ?              Mish-86          [-1, 128, 76, 76]               0
# ? Conv_Bn_Activation-87          [-1, 128, 76, 76]               0
# ?            Conv2d-88          [-1, 128, 76, 76]         147,456
# ?       BatchNorm2d-89          [-1, 128, 76, 76]             256
# ?              Mish-90          [-1, 128, 76, 76]               0
# ? Conv_Bn_Activation-91          [-1, 128, 76, 76]               0
# ?            Conv2d-92          [-1, 128, 76, 76]          16,384
# ?       BatchNorm2d-93          [-1, 128, 76, 76]             256
# ?              Mish-94          [-1, 128, 76, 76]               0
# ? Conv_Bn_Activation-95          [-1, 128, 76, 76]               0
# ?            Conv2d-96          [-1, 128, 76, 76]         147,456
# ?       BatchNorm2d-97          [-1, 128, 76, 76]             256
# ?              Mish-98          [-1, 128, 76, 76]               0
# ? Conv_Bn_Activation-99          [-1, 128, 76, 76]               0
# ?           Conv2d-100          [-1, 128, 76, 76]          16,384
# ?      BatchNorm2d-101          [-1, 128, 76, 76]             256
# ?             Mish-102          [-1, 128, 76, 76]               0
# ? Conv_Bn_Activation-103          [-1, 128, 76, 76]               0
# ?           Conv2d-104          [-1, 128, 76, 76]         147,456
# ?      BatchNorm2d-105          [-1, 128, 76, 76]             256
# ?             Mish-106          [-1, 128, 76, 76]               0
# ? Conv_Bn_Activation-107          [-1, 128, 76, 76]               0
# ?           Conv2d-108          [-1, 128, 76, 76]          16,384
# ?      BatchNorm2d-109          [-1, 128, 76, 76]             256
# ?             Mish-110          [-1, 128, 76, 76]               0
# ? Conv_Bn_Activation-111          [-1, 128, 76, 76]               0
# ?           Conv2d-112          [-1, 128, 76, 76]         147,456
# ?      BatchNorm2d-113          [-1, 128, 76, 76]             256
# ?             Mish-114          [-1, 128, 76, 76]               0
# ? Conv_Bn_Activation-115          [-1, 128, 76, 76]               0
# ?           Conv2d-116          [-1, 128, 76, 76]          16,384
# ?      BatchNorm2d-117          [-1, 128, 76, 76]             256
# ?             Mish-118          [-1, 128, 76, 76]               0
# ? Conv_Bn_Activation-119          [-1, 128, 76, 76]               0
# ?           Conv2d-120          [-1, 128, 76, 76]         147,456
# ?      BatchNorm2d-121          [-1, 128, 76, 76]             256
# ?             Mish-122          [-1, 128, 76, 76]               0
# ? Conv_Bn_Activation-123          [-1, 128, 76, 76]               0
# ?           Conv2d-124          [-1, 128, 76, 76]          16,384
# ?      BatchNorm2d-125          [-1, 128, 76, 76]             256
# ?             Mish-126          [-1, 128, 76, 76]               0
# ? Conv_Bn_Activation-127          [-1, 128, 76, 76]               0
# ?           Conv2d-128          [-1, 128, 76, 76]         147,456
# ?      BatchNorm2d-129          [-1, 128, 76, 76]             256
# ?             Mish-130          [-1, 128, 76, 76]               0
# ? Conv_Bn_Activation-131          [-1, 128, 76, 76]               0
# ?           Conv2d-132          [-1, 128, 76, 76]          16,384
# ?      BatchNorm2d-133          [-1, 128, 76, 76]             256
# ?             Mish-134          [-1, 128, 76, 76]               0
# ? Conv_Bn_Activation-135          [-1, 128, 76, 76]               0
# ?           Conv2d-136          [-1, 128, 76, 76]         147,456
# ?      BatchNorm2d-137          [-1, 128, 76, 76]             256
# ?             Mish-138          [-1, 128, 76, 76]               0
# ? Conv_Bn_Activation-139          [-1, 128, 76, 76]               0
# ?           Conv2d-140          [-1, 128, 76, 76]          16,384
# ?      BatchNorm2d-141          [-1, 128, 76, 76]             256
# ?             Mish-142          [-1, 128, 76, 76]               0
# ? Conv_Bn_Activation-143          [-1, 128, 76, 76]               0
# ?           Conv2d-144          [-1, 128, 76, 76]         147,456
# ?      BatchNorm2d-145          [-1, 128, 76, 76]             256
# ?             Mish-146          [-1, 128, 76, 76]               0
# ? Conv_Bn_Activation-147          [-1, 128, 76, 76]               0
# ?         ResBlock-148          [-1, 128, 76, 76]               0
# ?           Conv2d-149          [-1, 128, 76, 76]          16,384
# ?      BatchNorm2d-150          [-1, 128, 76, 76]             256
# ?             Mish-151          [-1, 128, 76, 76]               0
# ? Conv_Bn_Activation-152          [-1, 128, 76, 76]               0
# ?           Conv2d-153          [-1, 256, 76, 76]          65,536
# ?      BatchNorm2d-154          [-1, 256, 76, 76]             512
# ?             Mish-155          [-1, 256, 76, 76]               0
# ? Conv_Bn_Activation-156          [-1, 256, 76, 76]               0
# ?      DownSample3-157          [-1, 256, 76, 76]               0
# ?           Conv2d-158          [-1, 512, 38, 38]       1,179,648
# ?      BatchNorm2d-159          [-1, 512, 38, 38]           1,024
# ?             Mish-160          [-1, 512, 38, 38]               0
# ? Conv_Bn_Activation-161          [-1, 512, 38, 38]               0
# ?           Conv2d-162          [-1, 256, 38, 38]         131,072
# ?      BatchNorm2d-163          [-1, 256, 38, 38]             512
# ?             Mish-164          [-1, 256, 38, 38]               0
# ? Conv_Bn_Activation-165          [-1, 256, 38, 38]               0
# ?           Conv2d-166          [-1, 256, 38, 38]         131,072
# ?      BatchNorm2d-167          [-1, 256, 38, 38]             512
# ?             Mish-168          [-1, 256, 38, 38]               0
# ? Conv_Bn_Activation-169          [-1, 256, 38, 38]               0
# ?           Conv2d-170          [-1, 256, 38, 38]          65,536
# ?      BatchNorm2d-171          [-1, 256, 38, 38]             512
# ?             Mish-172          [-1, 256, 38, 38]               0
# ? Conv_Bn_Activation-173          [-1, 256, 38, 38]               0
# ?           Conv2d-174          [-1, 256, 38, 38]         589,824
# ?      BatchNorm2d-175          [-1, 256, 38, 38]             512
# ?             Mish-176          [-1, 256, 38, 38]               0
# ? Conv_Bn_Activation-177          [-1, 256, 38, 38]               0
# ?           Conv2d-178          [-1, 256, 38, 38]          65,536
# ?      BatchNorm2d-179          [-1, 256, 38, 38]             512
# ?             Mish-180          [-1, 256, 38, 38]               0
# ? Conv_Bn_Activation-181          [-1, 256, 38, 38]               0
# ?           Conv2d-182          [-1, 256, 38, 38]         589,824
# ?      BatchNorm2d-183          [-1, 256, 38, 38]             512
# ?             Mish-184          [-1, 256, 38, 38]               0
# ? Conv_Bn_Activation-185          [-1, 256, 38, 38]               0
# ?           Conv2d-186          [-1, 256, 38, 38]          65,536
# ?      BatchNorm2d-187          [-1, 256, 38, 38]             512
# ?             Mish-188          [-1, 256, 38, 38]               0
# ? Conv_Bn_Activation-189          [-1, 256, 38, 38]               0
# ?           Conv2d-190          [-1, 256, 38, 38]         589,824
# ?      BatchNorm2d-191          [-1, 256, 38, 38]             512
# ?             Mish-192          [-1, 256, 38, 38]               0
# ? Conv_Bn_Activation-193          [-1, 256, 38, 38]               0
# ?           Conv2d-194          [-1, 256, 38, 38]          65,536
# ?      BatchNorm2d-195          [-1, 256, 38, 38]             512
# ?             Mish-196          [-1, 256, 38, 38]               0
# ? Conv_Bn_Activation-197          [-1, 256, 38, 38]               0
# ?           Conv2d-198          [-1, 256, 38, 38]         589,824
# ?      BatchNorm2d-199          [-1, 256, 38, 38]             512
# ?             Mish-200          [-1, 256, 38, 38]               0
# ? Conv_Bn_Activation-201          [-1, 256, 38, 38]               0
# ?           Conv2d-202          [-1, 256, 38, 38]          65,536
# ?      BatchNorm2d-203          [-1, 256, 38, 38]             512
# ?             Mish-204          [-1, 256, 38, 38]               0
# ? Conv_Bn_Activation-205          [-1, 256, 38, 38]               0
# ?           Conv2d-206          [-1, 256, 38, 38]         589,824
# ?      BatchNorm2d-207          [-1, 256, 38, 38]             512
# ?             Mish-208          [-1, 256, 38, 38]               0
# ? Conv_Bn_Activation-209          [-1, 256, 38, 38]               0
# ?           Conv2d-210          [-1, 256, 38, 38]          65,536
# ?      BatchNorm2d-211          [-1, 256, 38, 38]             512
# ?             Mish-212          [-1, 256, 38, 38]               0
# ? Conv_Bn_Activation-213          [-1, 256, 38, 38]               0
# ?           Conv2d-214          [-1, 256, 38, 38]         589,824
# ?      BatchNorm2d-215          [-1, 256, 38, 38]             512
# ?             Mish-216          [-1, 256, 38, 38]               0
# ? Conv_Bn_Activation-217          [-1, 256, 38, 38]               0
# ?           Conv2d-218          [-1, 256, 38, 38]          65,536
# ?      BatchNorm2d-219          [-1, 256, 38, 38]             512
# ?             Mish-220          [-1, 256, 38, 38]               0
# ? Conv_Bn_Activation-221          [-1, 256, 38, 38]               0
# ?           Conv2d-222          [-1, 256, 38, 38]         589,824
# ?      BatchNorm2d-223          [-1, 256, 38, 38]             512
# ?             Mish-224          [-1, 256, 38, 38]               0
# ? Conv_Bn_Activation-225          [-1, 256, 38, 38]               0
# ?           Conv2d-226          [-1, 256, 38, 38]          65,536
# ?      BatchNorm2d-227          [-1, 256, 38, 38]             512
# ?             Mish-228          [-1, 256, 38, 38]               0
# ? Conv_Bn_Activation-229          [-1, 256, 38, 38]               0
# ?           Conv2d-230          [-1, 256, 38, 38]         589,824
# ?      BatchNorm2d-231          [-1, 256, 38, 38]             512
# ?             Mish-232          [-1, 256, 38, 38]               0
# ? Conv_Bn_Activation-233          [-1, 256, 38, 38]               0
# ?         ResBlock-234          [-1, 256, 38, 38]               0
# ?           Conv2d-235          [-1, 256, 38, 38]          65,536
# ?      BatchNorm2d-236          [-1, 256, 38, 38]             512
# ?             Mish-237          [-1, 256, 38, 38]               0
# ? Conv_Bn_Activation-238          [-1, 256, 38, 38]               0
# ?           Conv2d-239          [-1, 512, 38, 38]         262,144
# ?      BatchNorm2d-240          [-1, 512, 38, 38]           1,024
# ?             Mish-241          [-1, 512, 38, 38]               0
# ? Conv_Bn_Activation-242          [-1, 512, 38, 38]               0
# ?      DownSample4-243          [-1, 512, 38, 38]               0
# ?           Conv2d-244         [-1, 1024, 19, 19]       4,718,592
# ?      BatchNorm2d-245         [-1, 1024, 19, 19]           2,048
# ?             Mish-246         [-1, 1024, 19, 19]               0
# ? Conv_Bn_Activation-247         [-1, 1024, 19, 19]               0
# ?           Conv2d-248          [-1, 512, 19, 19]         524,288
# ?      BatchNorm2d-249          [-1, 512, 19, 19]           1,024
# ?             Mish-250          [-1, 512, 19, 19]               0
# ? Conv_Bn_Activation-251          [-1, 512, 19, 19]               0
# ?           Conv2d-252          [-1, 512, 19, 19]         524,288
# ?      BatchNorm2d-253          [-1, 512, 19, 19]           1,024
# ?             Mish-254          [-1, 512, 19, 19]               0
# ? Conv_Bn_Activation-255          [-1, 512, 19, 19]               0
# ?           Conv2d-256          [-1, 512, 19, 19]         262,144
# ?      BatchNorm2d-257          [-1, 512, 19, 19]           1,024
# ?             Mish-258          [-1, 512, 19, 19]               0
# ? Conv_Bn_Activation-259          [-1, 512, 19, 19]               0
# ?           Conv2d-260          [-1, 512, 19, 19]       2,359,296
# ?      BatchNorm2d-261          [-1, 512, 19, 19]           1,024
# ?             Mish-262          [-1, 512, 19, 19]               0
# ? Conv_Bn_Activation-263          [-1, 512, 19, 19]               0
# ?           Conv2d-264          [-1, 512, 19, 19]         262,144
# ?      BatchNorm2d-265          [-1, 512, 19, 19]           1,024
# ?             Mish-266          [-1, 512, 19, 19]               0
# ? Conv_Bn_Activation-267          [-1, 512, 19, 19]               0
# ?           Conv2d-268          [-1, 512, 19, 19]       2,359,296
# ?      BatchNorm2d-269          [-1, 512, 19, 19]           1,024
# ?             Mish-270          [-1, 512, 19, 19]               0
# ? Conv_Bn_Activation-271          [-1, 512, 19, 19]               0
# ?           Conv2d-272          [-1, 512, 19, 19]         262,144
# ?      BatchNorm2d-273          [-1, 512, 19, 19]           1,024
# ?             Mish-274          [-1, 512, 19, 19]               0
# ? Conv_Bn_Activation-275          [-1, 512, 19, 19]               0
# ?           Conv2d-276          [-1, 512, 19, 19]       2,359,296
# ?      BatchNorm2d-277          [-1, 512, 19, 19]           1,024
# ?             Mish-278          [-1, 512, 19, 19]               0
# ? Conv_Bn_Activation-279          [-1, 512, 19, 19]               0
# ?           Conv2d-280          [-1, 512, 19, 19]         262,144
# ?      BatchNorm2d-281          [-1, 512, 19, 19]           1,024
# ?             Mish-282          [-1, 512, 19, 19]               0
# ? Conv_Bn_Activation-283          [-1, 512, 19, 19]               0
# ?           Conv2d-284          [-1, 512, 19, 19]       2,359,296
# ?      BatchNorm2d-285          [-1, 512, 19, 19]           1,024
# ?             Mish-286          [-1, 512, 19, 19]               0
# ? Conv_Bn_Activation-287          [-1, 512, 19, 19]               0
# ?         ResBlock-288          [-1, 512, 19, 19]               0
# ?           Conv2d-289          [-1, 512, 19, 19]         262,144
# ?      BatchNorm2d-290          [-1, 512, 19, 19]           1,024
# ?             Mish-291          [-1, 512, 19, 19]               0
# ? Conv_Bn_Activation-292          [-1, 512, 19, 19]               0
# ?           Conv2d-293         [-1, 1024, 19, 19]       1,048,576
# ?      BatchNorm2d-294         [-1, 1024, 19, 19]           2,048
# ?             Mish-295         [-1, 1024, 19, 19]               0
# ? Conv_Bn_Activation-296         [-1, 1024, 19, 19]               0
# ?      DownSample5-297         [-1, 1024, 19, 19]               0
# ? ================================================================
# ? Total params: 26,617,184
# ? Trainable params: 26,617,184
# ? Non-trainable params: 0
# ? ----------------------------------------------------------------
# ? Input size (MB): 4.23
# ? Forward/backward pass size (MB): 3002.22
# ? Params size (MB): 101.54
# ? Estimated Total Size (MB): 3107.99
# ? ----------------------------------------------------------------

# ! YOLOv4 backbone: SPP, PAN (top-down path)
# ! Backbone으로부터 추출된 세 가지 scales의 feature maps (1024, 19, 19), (512, 38, 38), (256, 76, 76)을 입력한 결과:
# ? ----------------------------------------------------------------
# ?         Layer (type)               Output Shape         Param #
# ? ================================================================
# ?             Conv2d-1          [-1, 512, 19, 19]         524,288
# ?        BatchNorm2d-2          [-1, 512, 19, 19]           1,024
# ?          LeakyReLU-3          [-1, 512, 19, 19]               0
# ? Conv_Bn_Activation-4          [-1, 512, 19, 19]               0
# ?             Conv2d-5         [-1, 1024, 19, 19]       4,718,592
# ?        BatchNorm2d-6         [-1, 1024, 19, 19]           2,048
# ?          LeakyReLU-7         [-1, 1024, 19, 19]               0
# ? Conv_Bn_Activation-8         [-1, 1024, 19, 19]               0
# ?             Conv2d-9          [-1, 512, 19, 19]         524,288
# ?       BatchNorm2d-10          [-1, 512, 19, 19]           1,024
# ?         LeakyReLU-11          [-1, 512, 19, 19]               0
# ? Conv_Bn_Activation-12          [-1, 512, 19, 19]               0
# ?         MaxPool2d-13          [-1, 512, 19, 19]               0
# ?         MaxPool2d-14          [-1, 512, 19, 19]               0
# ?         MaxPool2d-15          [-1, 512, 19, 19]               0
# ?            Conv2d-16          [-1, 512, 19, 19]       1,048,576
# ?       BatchNorm2d-17          [-1, 512, 19, 19]           1,024
# ?         LeakyReLU-18          [-1, 512, 19, 19]               0
# ? Conv_Bn_Activation-19          [-1, 512, 19, 19]               0
# ?            Conv2d-20         [-1, 1024, 19, 19]       4,718,592
# ?       BatchNorm2d-21         [-1, 1024, 19, 19]           2,048
# ?         LeakyReLU-22         [-1, 1024, 19, 19]               0
# ? Conv_Bn_Activation-23         [-1, 1024, 19, 19]               0
# ?            Conv2d-24          [-1, 512, 19, 19]         524,288
# ?       BatchNorm2d-25          [-1, 512, 19, 19]           1,024
# ?         LeakyReLU-26          [-1, 512, 19, 19]               0
# ? Conv_Bn_Activation-27          [-1, 512, 19, 19]               0
# ?            Conv2d-28          [-1, 256, 19, 19]         131,072
# ?       BatchNorm2d-29          [-1, 256, 19, 19]             512
# ?         LeakyReLU-30          [-1, 256, 19, 19]               0
# ? Conv_Bn_Activation-31          [-1, 256, 19, 19]               0
# ?          Upsample-32          [-1, 256, 38, 38]               0
# ?            Conv2d-33          [-1, 256, 38, 38]         131,072
# ?       BatchNorm2d-34          [-1, 256, 38, 38]             512
# ?         LeakyReLU-35          [-1, 256, 38, 38]               0
# ? Conv_Bn_Activation-36          [-1, 256, 38, 38]               0
# ?            Conv2d-37          [-1, 256, 38, 38]         131,072
# ?       BatchNorm2d-38          [-1, 256, 38, 38]             512
# ?         LeakyReLU-39          [-1, 256, 38, 38]               0
# ? Conv_Bn_Activation-40          [-1, 256, 38, 38]               0
# ?            Conv2d-41          [-1, 512, 38, 38]       1,179,648
# ?       BatchNorm2d-42          [-1, 512, 38, 38]           1,024
# ?         LeakyReLU-43          [-1, 512, 38, 38]               0
# ? Conv_Bn_Activation-44          [-1, 512, 38, 38]               0
# ?            Conv2d-45          [-1, 256, 38, 38]         131,072
# ?       BatchNorm2d-46          [-1, 256, 38, 38]             512
# ?         LeakyReLU-47          [-1, 256, 38, 38]               0
# ? Conv_Bn_Activation-48          [-1, 256, 38, 38]               0
# ?            Conv2d-49          [-1, 512, 38, 38]       1,179,648
# ?       BatchNorm2d-50          [-1, 512, 38, 38]           1,024
# ?         LeakyReLU-51          [-1, 512, 38, 38]               0
# ? Conv_Bn_Activation-52          [-1, 512, 38, 38]               0
# ?            Conv2d-53          [-1, 256, 38, 38]         131,072
# ?       BatchNorm2d-54          [-1, 256, 38, 38]             512
# ?         LeakyReLU-55          [-1, 256, 38, 38]               0
# ? Conv_Bn_Activation-56          [-1, 256, 38, 38]               0
# ?            Conv2d-57          [-1, 128, 38, 38]          32,768
# ?       BatchNorm2d-58          [-1, 128, 38, 38]             256
# ?         LeakyReLU-59          [-1, 128, 38, 38]               0
# ? Conv_Bn_Activation-60          [-1, 128, 38, 38]               0
# ?          Upsample-61          [-1, 128, 76, 76]               0
# ?            Conv2d-62          [-1, 128, 76, 76]          32,768
# ?       BatchNorm2d-63          [-1, 128, 76, 76]             256
# ?         LeakyReLU-64          [-1, 128, 76, 76]               0
# ? Conv_Bn_Activation-65          [-1, 128, 76, 76]               0
# ?            Conv2d-66          [-1, 128, 76, 76]          32,768
# ?       BatchNorm2d-67          [-1, 128, 76, 76]             256
# ?         LeakyReLU-68          [-1, 128, 76, 76]               0
# ? Conv_Bn_Activation-69          [-1, 128, 76, 76]               0
# ?            Conv2d-70          [-1, 256, 76, 76]         294,912
# ?       BatchNorm2d-71          [-1, 256, 76, 76]             512
# ?         LeakyReLU-72          [-1, 256, 76, 76]               0
# ? Conv_Bn_Activation-73          [-1, 256, 76, 76]               0
# ?            Conv2d-74          [-1, 128, 76, 76]          32,768
# ?       BatchNorm2d-75          [-1, 128, 76, 76]             256
# ?         LeakyReLU-76          [-1, 128, 76, 76]               0
# ? Conv_Bn_Activation-77          [-1, 128, 76, 76]               0
# ?            Conv2d-78          [-1, 256, 76, 76]         294,912
# ?       BatchNorm2d-79          [-1, 256, 76, 76]             512
# ?         LeakyReLU-80          [-1, 256, 76, 76]               0
# ? Conv_Bn_Activation-81          [-1, 256, 76, 76]               0
# ?            Conv2d-82          [-1, 128, 76, 76]          32,768
# ?       BatchNorm2d-83          [-1, 128, 76, 76]             256
# ?         LeakyReLU-84          [-1, 128, 76, 76]               0
# ? Conv_Bn_Activation-85          [-1, 128, 76, 76]               0
# ? ================================================================
# ? Total params: 15,842,048
# ? Trainable params: 15,842,048
# ? Non-trainable params: 0
# ? ----------------------------------------------------------------
# ? Input size (MB): 1541599428608.00
# ? Forward/backward pass size (MB): 337.03
# ? Params size (MB): 60.43
# ? Estimated Total Size (MB): 1541599429005.46
# ? ----------------------------------------------------------------

# ! Head: YOLO-v3 head, PAN (bottom-up path)
# ! Neck에서 refined된 세 가지 scales의 feature maps (128, 76, 76), (256, 38, 38), (512, 19, 19)을 입력한 결과:
# ? ----------------------------------------------------------------
# ?         Layer (type)               Output Shape         Param #
# ? ================================================================
# ?             Conv2d-1          [-1, 256, 76, 76]         294,912
# ?        BatchNorm2d-2          [-1, 256, 76, 76]             512
# ?          LeakyReLU-3          [-1, 256, 76, 76]               0
# ? Conv_Bn_Activation-4          [-1, 256, 76, 76]               0
# ?             Conv2d-5          [-1, 255, 76, 76]          65,535
# ? Conv_Bn_Activation-6          [-1, 255, 76, 76]               0
# ?             Conv2d-7          [-1, 256, 38, 38]         294,912
# ?        BatchNorm2d-8          [-1, 256, 38, 38]             512
# ?          LeakyReLU-9          [-1, 256, 38, 38]               0
# ? Conv_Bn_Activation-10          [-1, 256, 38, 38]               0
# ?            Conv2d-11          [-1, 256, 38, 38]         131,072
# ?       BatchNorm2d-12          [-1, 256, 38, 38]             512
# ?         LeakyReLU-13          [-1, 256, 38, 38]               0
# ? Conv_Bn_Activation-14          [-1, 256, 38, 38]               0
# ?            Conv2d-15          [-1, 512, 38, 38]       1,179,648
# ?       BatchNorm2d-16          [-1, 512, 38, 38]           1,024
# ?         LeakyReLU-17          [-1, 512, 38, 38]               0
# ? Conv_Bn_Activation-18          [-1, 512, 38, 38]               0
# ?            Conv2d-19          [-1, 256, 38, 38]         131,072
# ?       BatchNorm2d-20          [-1, 256, 38, 38]             512
# ?         LeakyReLU-21          [-1, 256, 38, 38]               0
# ? Conv_Bn_Activation-22          [-1, 256, 38, 38]               0
# ?            Conv2d-23          [-1, 512, 38, 38]       1,179,648
# ?       BatchNorm2d-24          [-1, 512, 38, 38]           1,024
# ?         LeakyReLU-25          [-1, 512, 38, 38]               0
# ? Conv_Bn_Activation-26          [-1, 512, 38, 38]               0
# ?            Conv2d-27          [-1, 256, 38, 38]         131,072
# ?       BatchNorm2d-28          [-1, 256, 38, 38]             512
# ?         LeakyReLU-29          [-1, 256, 38, 38]               0
# ? Conv_Bn_Activation-30          [-1, 256, 38, 38]               0
# ?            Conv2d-31          [-1, 512, 38, 38]       1,179,648
# ?       BatchNorm2d-32          [-1, 512, 38, 38]           1,024
# ?         LeakyReLU-33          [-1, 512, 38, 38]               0
# ? Conv_Bn_Activation-34          [-1, 512, 38, 38]               0
# ?            Conv2d-35          [-1, 255, 38, 38]         130,815
# ? Conv_Bn_Activation-36          [-1, 255, 38, 38]               0
# ?            Conv2d-37          [-1, 512, 19, 19]       1,179,648
# ?       BatchNorm2d-38          [-1, 512, 19, 19]           1,024
# ?         LeakyReLU-39          [-1, 512, 19, 19]               0
# ? Conv_Bn_Activation-40          [-1, 512, 19, 19]               0
# ?            Conv2d-41          [-1, 512, 19, 19]         524,288
# ?       BatchNorm2d-42          [-1, 512, 19, 19]           1,024
# ?         LeakyReLU-43          [-1, 512, 19, 19]               0
# ? Conv_Bn_Activation-44          [-1, 512, 19, 19]               0
# ?            Conv2d-45         [-1, 1024, 19, 19]       4,718,592
# ?       BatchNorm2d-46         [-1, 1024, 19, 19]           2,048
# ?         LeakyReLU-47         [-1, 1024, 19, 19]               0
# ? Conv_Bn_Activation-48         [-1, 1024, 19, 19]               0
# ?            Conv2d-49          [-1, 512, 19, 19]         524,288
# ?       BatchNorm2d-50          [-1, 512, 19, 19]           1,024
# ?         LeakyReLU-51          [-1, 512, 19, 19]               0
# ? Conv_Bn_Activation-52          [-1, 512, 19, 19]               0
# ?            Conv2d-53         [-1, 1024, 19, 19]       4,718,592
# ?       BatchNorm2d-54         [-1, 1024, 19, 19]           2,048
# ?         LeakyReLU-55         [-1, 1024, 19, 19]               0
# ? Conv_Bn_Activation-56         [-1, 1024, 19, 19]               0
# ?            Conv2d-57          [-1, 512, 19, 19]         524,288
# ?       BatchNorm2d-58          [-1, 512, 19, 19]           1,024
# ?         LeakyReLU-59          [-1, 512, 19, 19]               0
# ? Conv_Bn_Activation-60          [-1, 512, 19, 19]               0
# ?            Conv2d-61         [-1, 1024, 19, 19]       4,718,592
# ?       BatchNorm2d-62         [-1, 1024, 19, 19]           2,048
# ?         LeakyReLU-63         [-1, 1024, 19, 19]               0
# ? Conv_Bn_Activation-64         [-1, 1024, 19, 19]               0
# ?            Conv2d-65          [-1, 255, 19, 19]         261,375
# ? Conv_Bn_Activation-66          [-1, 255, 19, 19]               0
# ? ================================================================
# ? Total params: 21,903,869
# ? Trainable params: 21,903,869
# ? Non-trainable params: 0
# ? ----------------------------------------------------------------
# ? Input size (MB): 192699928576.00
# ? Forward/backward pass size (MB): 243.84
# ? Params size (MB): 83.56
# ? Estimated Total Size (MB): 192699928903.40
# ? ----------------------------------------------------------------