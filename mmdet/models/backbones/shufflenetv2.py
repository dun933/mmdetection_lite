# -*- coding: utf-8 -*-
# @Time    : 2019/11/1 15:31
# @Author  : zhoujun

import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from ..registry import BACKBONES
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo

from mmdet.ops import (ContextBlock, GeneralizedAttention, build_conv_layer,
                       build_norm_layer)
from mmdet.utils import get_root_logger

model_urls = {
    '0.5x': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
    '1.0x': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
    '1.5x': None,
    '2.0x': None,
}


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out

@BACKBONES.register_module
class ShuffleNetV2(nn.Module):
    def __init__(self, model_size = "1.0x",
                 num_stages=4,
                 norm_eval=True,
                 frozen_stages=-1,
                 ):
        super(ShuffleNetV2, self).__init__()
        self.model_size = model_size
        self.num_stages = num_stages
        self.norm_eval = norm_eval
        self.frozen_stages = frozen_stages
        stages_repeats = [4,8,4]
        if model_size == "0.5x":
            stages_out_channels =  [24, 48, 96, 192, 1024]
        elif model_size == "1.0x":
            stages_out_channels =  [24, 116, 232, 464, 1024]

        else:
            raise  ("not support type {}".format(type))

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

    def load_model(self, state_dict):
        new_model = self.state_dict()
        new_keys = list(new_model.keys())
        old_keys = list(state_dict.keys())
        restore_dict = OrderedDict()
        for id in range(len(new_keys)):
            restore_dict[new_keys[id]] = state_dict[old_keys[id]]
        self.load_state_dict(restore_dict)

    def forward(self, x):
        x = self.conv1(x)
        c2 = self.maxpool(x)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        # c5 = self.conv5(c5)
        if self.num_stages == 2 :
            return c4, c5
        elif self.num_stages == 3:
            return c3, c4, c5
        elif self.num_stages == 4:
            return c2, c3, c4, c5

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for m in [self.conv1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(2, self.frozen_stages + 2):
            m = getattr(self, 'stage{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    # def init_weights(self, pretrained=None):
    #     # from collections import OrderedDict
    #     # temp = OrderedDict()
    #
    #     if isinstance(pretrained, str):
    #         logger = get_root_logger()
    #
    #         load_checkpoint(self, pretrained, strict=False, logger=logger)
    #
    #     elif pretrained is None:
    #         for m in self.modules():
    #
    #
    #             if isinstance(m, nn.Conv2d):
    #                 kaiming_init(m)
    #             elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
    #                 constant_init(m, 1)
    #     else:
    #         raise TypeError('pretrained must be a str or None')

    def init_weights(self, pretrained=None):
        # from collections import OrderedDict
        # temp = OrderedDict()

        if isinstance(pretrained, str):

            state_dict = model_zoo.load_url(pretrained,
                                            progress=True)
            self.load_model(state_dict)

        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')
    def train(self, mode=True):
        super(ShuffleNetV2, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()