from collections import OrderedDict

import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F

from torch.jit.annotations import Tuple, List, Dict

# AC-FPN 实现代码

class DenseBlock(nn.Module):
    def __init__(self, input_num, num1, num2, rate, drop_out):
        super(DenseBlock, self).__init__()

        # C: 2048 --> 512 --> 256
        self.conv1x1 = nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1)
        self.ConvGN = nn.GroupNorm(num_groups=32, num_channels=num1)
        self.relu1 = nn.ReLU(inplace=True)
        self.dilaconv = nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3, padding=1 * rate, dilation=rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = self.ConvGN(self.conv1x1(x))
        x = self.relu1(x)
        x = self.dilaconv(x)
        x = self.relu2(x)
        x = self.drop(x)
        return x


class DenseAPP(nn.Module):
    def __init__(self, num_channels=2048):
        super(DenseAPP, self).__init__()
        self.drop_out = 0.1
        self.channels1 = 512
        self.channels2 = 256
        self.num_channels = num_channels
        self.aspp3 = DenseBlock(self.num_channels, num1=self.channels1, num2=self.channels2, rate=3,
                                drop_out=self.drop_out)
        self.aspp6 = DenseBlock(self.num_channels + self.channels2 * 1, num1=self.channels1, num2=self.channels2,
                                rate=6,
                                drop_out=self.drop_out)
        self.aspp12 = DenseBlock(self.num_channels + self.channels2 * 2, num1=self.channels1, num2=self.channels2,
                                 rate=12,
                                 drop_out=self.drop_out)
        self.aspp18 = DenseBlock(self.num_channels + self.channels2 * 3, num1=self.channels1, num2=self.channels2,
                                 rate=18,
                                 drop_out=self.drop_out)
        self.aspp24 = DenseBlock(self.num_channels + self.channels2 * 4, num1=self.channels1, num2=self.channels2,
                                 rate=24,
                                 drop_out=self.drop_out)
        self.conv1x1 = nn.Conv2d(in_channels=5*self.channels2, out_channels=256, kernel_size=1)
        self.ConvGN = nn.GroupNorm(num_groups=32, num_channels=256)

    def forward(self, feature):
        aspp3 = self.aspp3(feature)
        feature = torch.concat((aspp3, feature), dim=1)
        aspp6 = self.aspp6(feature)
        feature = torch.concat((aspp6, feature), dim=1)
        aspp12 = self.aspp12(feature)
        feature = torch.concat((aspp12, feature), dim=1)
        aspp18 = self.aspp18(feature)
        feature = torch.concat((aspp18, feature), dim=1)
        aspp24 = self.aspp24(feature)

        x = torch.concat((aspp3, aspp6, aspp12, aspp18, aspp24), dim=1)
        out = self.ConvGN(self.conv1x1(x))
        return out


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels, extra_blocks=None):
        super().__init__()
        self.dense = DenseAPP(num_channels=in_channels_list[-1])
        # 用来调整resnet特征矩阵(layer1,2,3,4)的channel（kernel_size=1）
        self.inner_blocks = nn.ModuleList()
        # 对调整后的特征矩阵使用3x3的卷积核来得到对应的预测特征矩阵
        self.layer_blocks = nn.ModuleList()
        for in_channels in in_channels_list:
            if in_channels == 0:
                continue
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

        # initialize parameters now to avoid modifying the initialization of top_blocks
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

        self.extra_blocks = extra_blocks

    def get_result_from_inner_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.inner_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.inner_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def get_result_from_layer_blocks(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.layer_blocks[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.layer_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def forward(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Computes the FPN for a set of feature maps.
        Arguments:
            x (OrderedDict[Tensor]): feature maps for each feature level.
        Returns:
            results (OrderedDict[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """
        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())
        # 将C5送入DenseAPP中获得上下文信息
        dense = self.dense(x[-1])
        # 将resnet layer4的channel调整到指定的out_channels
        # last_inner = self.inner_blocks[-1](x[-1])
        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        # result中保存着每个预测特征层
        results = []
        # 将layer4调整channel后的特征矩阵，通过3x3卷积后得到对应的预测特征矩阵
        # results.append(self.layer_blocks[-1](last_inner))

        P5 = dense + self.get_result_from_layer_blocks(last_inner, -1)
        results.append(P5)

        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = F.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

        # 在layer4对应的预测特征层基础上生成预测特征矩阵5
        if self.extra_blocks is not None:
            results, names = self.extra_blocks(results, x, names)

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out


class LastLevelMaxPool(torch.nn.Module):
    """
    Applies a max_pool2d on top of the last feature map
    """

    def forward(self, x: List[Tensor], y: List[Tensor], names: List[str]) -> Tuple[List[Tensor], List[str]]:
        names.append("pool")
        x.append(F.max_pool2d(x[-1], 1, 2, 0))  # input, kernel_size, stride, padding
        return x, names
