from mmseg.registry import MODELS
from mmengine.model import BaseModule
from torch import nn as nn
from torch.nn import functional as F
from typing import Callable, Optional
from torch import Tensor
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block as TransformerBlock


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


@MODELS.register_module()
class FMM(BaseModule):
    def __init__(
        self,
        in_channels,
        rank_dim=4,
        mlp_nums=1,
        model_type="mlp",
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        qk_norm=False,
        init_values=None,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        self.adapters = nn.ModuleList()
        if model_type == "mlp":
            for in_channel in in_channels:
                mlp_list = []
                for _ in range(mlp_nums):
                    mlp_list.append(
                        Mlp(
                            in_channel,
                            hidden_features=in_channel // rank_dim,
                            out_features=in_channel,
                        )
                    )
                mlp_model = nn.Sequential(*mlp_list)
                self.adapters.append(mlp_model)

        elif model_type == "vitBlock":
            for in_channel in in_channels:
                model_list = []
                for _ in range(mlp_nums):
                    model_list.append(
                        TransformerBlock(
                            in_channel,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_norm=qk_norm,
                            init_values=init_values,
                            proj_drop=proj_drop_rate,
                            attn_drop=attn_drop_rate,
                        )
                    )
                self.adapters.append(nn.Sequential(*model_list))
        
        else:
            raise ValueError(f"model type must in ['mlp','vitBlock'],actually is {model_type}")

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        outs = []
        for index, x in enumerate(inputs):
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1)
            x = x.reshape(B, -1, C)
            x = self.adapters[index](x)
            x = x.reshape(B, H, W, C)
            x = x.permute(0, 3, 1, 2)
            outs.append(x)
        return tuple(outs)
