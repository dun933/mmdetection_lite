import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

from mmdet.core import auto_fp16
from mmdet.ops import ConvModule
from ..registry import NECKS

@NECKS.register_module
class CEM(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                            ):
        super(CEM, self).__init__()

        self.in_channels = in_channels
        self.conv4 = nn.Conv2d(in_channels[0], out_channels, 1, bias=True)
        self.conv5 = nn.Conv2d(in_channels[1], out_channels, 1, bias=True)
        self.convlast = nn.Conv2d(in_channels[1], out_channels, 1, bias=True)
        self.unsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.init_weights()
    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        C4_lat = self.conv4(inputs[-2])
        C5_lat = self.conv5(inputs[-1])
        C5_lat = self.unsample(C5_lat)
        avg_pool = self.avg_pool(inputs[-1])
        Cglb_lat = self.convlast(avg_pool)

        outs = [C4_lat + C5_lat + Cglb_lat]


        return tuple(outs)
