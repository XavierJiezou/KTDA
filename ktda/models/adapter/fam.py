from mmseg.registry import MODELS
from mmengine.model import BaseModule
from torch import nn as nn
from torch.nn import functional as F
from timm.models.layers import trunc_normal_


@MODELS.register_module()
class FAM(BaseModule):
    def __init__(self, in_channels, out_channels, output_size,init_cfg=None):
        super().__init__(init_cfg)
        self.convert = nn.ModuleList()
        self.output_size = output_size
        if isinstance(out_channels, int):
            out_channels = [out_channels] * len(in_channels)
        for in_channel, out_channel in zip(in_channels, out_channels):
            self.convert.append(
                nn.Conv2d(in_channel, out_channel, kernel_size=1),
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
        

    def forward(self, inputs):
        outs = []
        for index, x in enumerate(inputs):
            x = self.convert[index](x)
            x = F.interpolate(
                x, size=(self.output_size,self.output_size), align_corners=False, mode="bilinear"
            )
            outs.append(x)
        return tuple(outs)
