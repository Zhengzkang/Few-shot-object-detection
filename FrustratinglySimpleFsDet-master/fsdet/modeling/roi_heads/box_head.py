import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.utils.registry import Registry

ROI_BOX_HEAD_REGISTRY = Registry("ROI_BOX_HEAD")
ROI_BOX_HEAD_REGISTRY.__doc__ = """
Registry for box heads, which make box predictions from per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


@ROI_BOX_HEAD_REGISTRY.register()
class FastRCNNConvFCHead(nn.Module):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and
    several fc layers (each followed by relu).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            num_conv, num_fc: the number of conv/fc layers
            conv_dim/fc_dim: the dimension of the conv/fc layers
            norm: normalization for the conv layers
        """
        super().__init__()

        # fmt: off
        num_conv   = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv_dim   = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc     = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim     = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        norm       = cfg.MODEL.ROI_BOX_HEAD.NORM
        # fmt: on
        assert num_conv + num_fc > 0

        self._output_size = (
            input_shape.channels,
            input_shape.height,
            input_shape.width,
        )

        self.conv_norm_relus = []
        for k in range(num_conv):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not norm,
                norm=get_norm(norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (
                conv_dim,
                self._output_size[1],
                self._output_size[2],
            )

        self.fcs = []
        for k in range(num_fc):
            fc = nn.Linear(np.prod(self._output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for layer in self.fcs:
                x = F.relu(layer(x))
        return x

    @property
    def output_size(self):
        return self._output_size


@ROI_BOX_HEAD_REGISTRY.register()
class FastRCNNConvFCMultiHead(nn.Module):
    """
        A multi-head with several 3x3 conv layers (each followed by norm & relu) and
        several fc layers (each followed by relu). Some fc layers may be redundant, depending on split_at_fc and
        num_heads.
        """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        We make following assumptions:
            1. All conv layers are placed before the first fc layer
            2. We don't allow splitting the head at conv layers, for simplicity
        Arguments:
            cfg:
            input_shape:
            split_at_fc: id of fc layer to start splitting the head. Ids start by 1. Per default, we use two fc layers,
            starting to split at fc2 (which is the second fc layer)
            num_heads: number of parallel roi heads
        The following attributes are parsed from config:
            num_conv, num_fc: the number of conv/fc layers
            conv_dim/fc_dim: the dimension of the conv/fc layers
            norm: normalization for the conv layers
        """
        super().__init__()

        # fmt: off
        num_conv = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        norm = cfg.MODEL.ROI_BOX_HEAD.NORM
        num_heads = cfg.MODEL.ROI_BOX_HEAD.NUM_HEADS  # number of parallel roi heads
        # id of fc layer to start splitting the head. Ids start by 1. Per default, we use two fc layers, starting to
        #  split at fc2 (which is the second fc layer)
        self.split_at_fc = cfg.MODEL.ROI_BOX_HEAD.SPLIT_AT_FC
        # fmt: on
        assert num_heads > 0
        assert num_heads == 1 or num_fc > 0  # multi-head without fcs is not allowed!
        assert num_fc + num_conv > 0
        assert self.split_at_fc <= num_fc

        self._output_size = (
            input_shape.channels,
            input_shape.height,
            input_shape.width,
        )

        self.conv_norm_relus = []
        for k in range(num_conv):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not norm,
                norm=get_norm(norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (
                conv_dim,
                self._output_size[1],
                self._output_size[2],
            )

        self.fcs = []
        for k in range(1, num_fc + 1):
            if k >= self.split_at_fc and num_heads > 1:
                tmp_fcs = []
                for i in range(1, num_heads + 1):
                    fc = nn.Linear(np.prod(self._output_size), fc_dim)
                    self.add_module("fc{}:{}".format(k, i), fc)  # '.' is not allowed!
                    tmp_fcs.append(fc)
                self.fcs.append(tmp_fcs)
                self._output_size = num_heads * fc_dim
            else:
                fc = nn.Linear(np.prod(self._output_size), fc_dim)
                self.add_module("fc{}".format(k), fc)
                self.fcs.append([fc])
                self._output_size = fc_dim

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layers in self.fcs:
            for layer in layers:
                weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            inputs = [x]  # inputs for first fc layers.
            for layers in self.fcs:
                assert len(layers)  # need at least one fc layer in the list
                if len(layers) > len(inputs):
                    assert len(layers) % len(inputs) == 0
                    inputs = (len(layers) // len(inputs)) * inputs
                outputs = []
                for i, layer in enumerate(layers):
                    # TODO: sequential forward of parallelisable branch could be slow!
                    outputs.append(F.relu(layer(inputs[i])))
                inputs = outputs
            return outputs
        else:
            return x

    @property
    def output_size(self):
        return self._output_size


def build_box_head(cfg, input_shape):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_BOX_HEAD.NAME
    return ROI_BOX_HEAD_REGISTRY.get(name)(cfg, input_shape)
