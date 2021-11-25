# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pdb
import paddle
from paddle.fluid.layers.nn import pad
import paddle.nn as nn
from paddle.nn import layer
import paddle.nn.functional as F

from paddleseg.models import layers
from paddleseg.cvlibs import manager
from paddleseg.utils import utils


@manager.MODELS.add_component
class ENCNet(nn.Layer):
    """
    Context Encoding for Semantic Segmentation based on PaddlePaddle.

    The original article refers to
    , et al. "Context Encoding for Semantic Segmentation"
    (https://arxiv.org/abs/1803.08904)

    Args:
        num_classes (int): The unique number of target classes.
        backbone (Paddle.nn.Layer): A backbone network.
        backbone_indices (tuple): The values in the tuple indicate the indices of output of backbone.
        in_channels (int): Encoding module channels.
        num_codes (int): Number of codes.
        use_se_loss (bool): Whether use se loss or not
        stage_num (int): The iteration number for EM.
        momentum (float): The parameter for updating bases.
        concat_input (bool): Whether concat the input and output of convs before classification layer. Default: True
        enable_auxiliary_loss (bool, optional): A bool value indicates whether adding auxiliary loss. Default: True.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.  Default: False.
        pretrained (str, optional): The path or url of pretrained model. Default: None.
    """

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=(1, 2, 3),
                 channels=512,
                 num_codes=32,
                 use_se_loss=True,
                 add_lateral=True,
                 enable_se_loss=True,
                 align_corners=False,
                 enable_auxiliary_loss=True,
                 pretrained=None):
        super().__init__()
        self.backbone = backbone
        self.backbone_indices = backbone_indices
        in_channels = [self.backbone.feat_channels[i] for i in backbone_indices]
        self.use_se_loss = use_se_loss
        self.head = EncHead(num_classes, in_channels, channels, num_codes, add_lateral, enable_se_loss, enable_auxiliary_loss, align_corners=align_corners)

        self.align_corners = align_corners
        self.pretrained = pretrained
        self.init_weight()


    def forward(self, x):
        feats = self.backbone(x)
        feats = [feats[i] for i in self.backbone_indices]
        logit_list = self.head(feats)
        if self.use_se_loss and self.training:
            se_output = logit_list[-1]
            logit_list = logit_list[:-1]
        logit_list = [
            F.interpolate(
                logit,
                paddle.shape(x)[2:],
                mode='bilinear',
                align_corners=self.align_corners) for logit in logit_list
        ]
        if self.use_se_loss and self.training:
            logit_list.append(se_output)
        return logit_list

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

class EncHead(nn.Layer):
    """
    The EncHead head.

    Args:
        num_classes (int): The unique number of target classes.
        in_channels (tuple): The number of input channels.
        num_codes (int): Number of codes.
        enable_auxiliary_loss (bool, optional): A bool value indicates whether adding auxiliary loss. Default: True.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 channels,
                 num_codes,
                 add_lateral=True,
                 enable_se_loss=True,
                 enable_auxiliary_loss = True,
                 align_corners=False):
        super(EncHead, self).__init__()
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.add_lateral = add_lateral
        self.channels = channels
        self.enable_se_loss = enable_se_loss
        self.num_codes = num_codes

        self.encm = EncModule(channels, num_codes)
        self.enabel_aux = enable_auxiliary_loss
        self.aux = nn.Sequential(
            layers.ConvBNReLU(
                in_channels=in_channels[-2], out_channels=256, kernel_size=3),
            nn.Dropout2D(p=0.1), nn.Conv2D(256, num_classes, 1))
        self.bottleneck = layers.ConvBNReLU(
            in_channels=self.in_channels[-1], out_channels=channels, kernel_size=3)
        
        if add_lateral:
            self.lateral_convs = nn.LayerList()
            for in_channel in self.in_channels[:-1]:
                self.lateral_convs.append(layers.ConvBNReLU(in_channel, self.channels, 1))
            self.fusion = layers.ConvBNReLU(len(self.in_channels) * channels, channels, 3)
        
        if enable_se_loss:
            self.se_output = nn.Linear(channels, num_classes)

        self.cls = nn.Sequential(nn.Conv2D(channels, num_classes, 1))


    def forward(self, feat_list):
        feat = self.bottleneck(feat_list[-1])
        if self.add_lateral:
            laterals = [
                F.interpolate(
                    lateral_conv(feat_list[i]),
                    size=feat.shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners)
                for i, lateral_conv in enumerate(self.lateral_convs)
            ]
            feat = self.fusion(paddle.concat([feat, *laterals], 1))
        encode_feat, output = self.encm(feat)
        output_list = [output]
        if self.enabel_aux and self.training:
            aux = self.aux(feat_list[-2])
            output_list.append(aux)
        if self.enable_se_loss and self.training:
            se_out = self.se_output(encode_feat)
            output_list.append(se_out)
        return output_list


class EncModule(nn.Layer):
    """Encoding Module used in EncNet.
    Args:
        in_channels (int): Input channels.
        num_codes (int): Number of code words.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
    """

    def __init__(self, in_channels, num_codes):
        super(EncModule, self).__init__()
        self.encoding_project = layers.ConvBNReLU(
            in_channels,
            in_channels,
            1)

        # if norm_cfg is not None:
        #     encoding_norm_cfg = norm_cfg.copy()
        #     if encoding_norm_cfg['type'] in ['BN', 'IN']:
        #         encoding_norm_cfg['type'] += '1d'
        #     else:
        #         encoding_norm_cfg['type'] = encoding_norm_cfg['type'].replace(
        #             '2d', '1d')
        # else:
        #     # fallback to BN1d
        #     encoding_norm_cfg = dict(type='BN1d')
        self.encoding = nn.Sequential(
            Encoding(channels=in_channels, num_codes=num_codes),
            layers.SyncBatchNorm(num_codes),
            nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels), nn.Sigmoid())

    def forward(self, x):
        encoding_projection = self.encoding_project(x)
        
        encoding_feat = self.encoding(encoding_projection)
        encoding_feat = encoding_feat.squeeze(2)
        #batch, num_codes, channels
        encoding_feat = encoding_feat.mean(1)
        batch_size, channels, _, _ = x.shape
        gamma = self.fc(encoding_feat)
        y = paddle.reshape(gamma, [batch_size, channels, 1, 1])
        output = F.relu_(x + x * y)
        return encoding_feat, output

class Encoding(nn.Layer):
    """Encoding Layer: a learnable residual encoder.
    Input is of shape  (batch_size, channels, height, width).
    Output is of shape (batch_size, num_codes, channels).
    Args:
        channels: dimension of the features or feature channels
        num_codes: number of code words
    """

    def __init__(self, channels, num_codes):
        super(Encoding, self).__init__()
        # init codewords and smoothing factor
        self.channels, self.num_codes = channels, num_codes
        std = 1. / ((num_codes * channels)**0.5)
        # [num_codes, channels]
        
        self.codewords = self.create_parameter(shape=[num_codes, channels], dtype='float32', default_initializer=nn.initializer.Uniform(-std, std))
        # self.codewords = nn.Parameter(
        #     paddle.empty(num_codes, channels,
        #                 dtype=paddle.float).uniform_(-std, std),
        #     requires_grad=True)
        # [num_codes]
        self.scale = self.create_parameter(shape=[num_codes], dtype='float32', default_initializer=nn.initializer.Uniform(-1, 0))
        # self.scale = nn.Parameter(
        #     paddle.empty(num_codes, dtype=paddle.float).uniform_(-1, 0),
        #     requires_grad=True)

    @staticmethod
    def scaled_l2(x, codewords, scale):
        num_codes, channels = codewords.shape
        batch_size = x.shape[0]
        reshaped_scale = paddle.reshape(scale, [1, 1, num_codes])
        channel = x.shape[1]
        expanded_x = x.unsqueeze(2).expand((batch_size, channel, num_codes, channels))
        reshaped_codewords = paddle.reshape(codewords, [1, 1, num_codes, channels])

        scaled_l2_norm = reshaped_scale * (
            expanded_x - reshaped_codewords).pow(2).sum(3)
        return scaled_l2_norm

    @staticmethod
    def aggregate(assignment_weights, x, codewords):
        num_codes, channels = codewords.shape
        reshaped_codewords =  paddle.reshape(codewords, [1, 1, num_codes, channels])
        batch_size = x.shape[0]
        channel = x.shape[1]
        expanded_x = x.unsqueeze(2).expand((batch_size, channel , num_codes, channels))
        encoded_feat = (assignment_weights.unsqueeze(3) * (expanded_x - reshaped_codewords)).sum(1)
        return encoded_feat

    def forward(self, x):
        assert x.dim() == 4 and x.shape[1] == self.channels
        # [batch_size, channels, height, width]
        batch_size = x.shape[0]
        # [batch_size, height x width, channels]
        x = paddle.reshape(x, [batch_size, self.channels, -1]).transpose([0, 2, 1])
        
        # assignment_weights: [batch_size, channels, num_codes]
        assignment_weights = F.softmax(
            self.scaled_l2(x, self.codewords, self.scale), axis=2)
        # aggregate
        encoded_feat = self.aggregate(assignment_weights, x, self.codewords)
        encoded_feat = encoded_feat.unsqueeze(2)
        return encoded_feat

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(Nx{self.channels}xHxW =>Nx{self.num_codes}' \
                    f'x{self.channels})'
        return repr_str
