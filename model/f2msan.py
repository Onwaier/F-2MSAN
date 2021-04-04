from torch import nn
import torch
import torchvision
from torch.nn import functional as F

import math

def conv3x3(_in, _out):
    return nn.Conv2d(_in, _out, kernel_size=3, stride=1, padding=1)

def conv3x3_2(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ConvRelu(nn.Module):
    def __init__(self, _in, _out):
        super().__init__()
        self.conv = conv3x3(_in, _out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3_2(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3_2(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(DecoderBlockV2, self).__init__()
        self.in_channels = in_channels
        self.up = F.interpolate
        self.cr1 = ConvRelu(in_channels, middle_channels)
        self.cr2 = ConvRelu(middle_channels, out_channels)

    def forward(self, x):
        x = self.up(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.cr2(self.cr1(x))
        return x

# Facial Attention Network
class FAN(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, num_filters=32, encoder_depth=34, pretrained=False):
        super(FAN, self).__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.num_filters = num_filters

        if encoder_depth == 34:
            self.encoder = torchvision.models.resnet34(pretrained=pretrained)
            self.encoder.requires_grad_(False)
            bottom_channel_nr = 512
        elif encoder_depth == 101:
            self.encoder = torchvision.models.resnet101(pretrained=pretrained)
            bottom_channel_nr = 2048
        elif encoder_depth == 152:
            self.encoder = torchvision.models.resnet152(pretrained=pretrained)
            bottom_channel_nr = 2048
        else:
            raise NotImplementedError('only 34, 101, 152 version of Resnet are implemented')

        # attention module
        self.pool = nn.MaxPool2d(2, 2)
        # self.relu  = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu, self.pool)
        self.conv2 = self.encoder.layer1
        self.conv3 = self.encoder.layer2
        self.conv4 = self.encoder.layer3
        self.conv5 = self.encoder.layer4

        self.center = DecoderBlockV2(bottom_channel_nr, num_filters * 8 * 2, num_filters * 8)
        self.dec5 = DecoderBlockV2(bottom_channel_nr + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderBlockV2(bottom_channel_nr // 2 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec3 = DecoderBlockV2(bottom_channel_nr // 4 + num_filters * 8, num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderBlockV2(bottom_channel_nr // 8 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters)

        self.attention_map = nn.Sequential(
            ConvRelu(num_filters, num_filters),
            nn.Conv2d(num_filters, 1, kernel_size=1)
        )

    def forward(self, x):

        # attention module
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        pool = self.pool(conv5)
        center = self.center(pool)
        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)

        # attention map
        x = self.attention_map(dec1)
        return x

# Multi Self Attention Network
class MSAN(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(MSAN, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.bn2 = nn.LayerNorm(512)
        # self.bn3 = nn.LayerNorm(1024) # 原来1024
        # self.bn4 = nn.LayerNorm(1536)
        self.alpha = nn.Sequential(
            nn.Linear(512, 1),
            # nn.Dropout(p=0.5),
            nn.Sigmoid())
        self.beta = nn.Sequential(
            nn.Linear(1024, 1),
            # nn.Dropout(p=0.5),
            nn.Sigmoid())

        self.delta = nn.Sequential(
            nn.Linear(1536, 1),
            # nn.Dropout(p=0.5),
            nn.Sigmoid())
        # self.fc = nn.Linear(1024, 7)
        self.fc = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(1536, 7)  # 原来1024
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print 'input image shape',x.shape
        vs = []
        vs2 = []
        alphas = []
        for i in range(6):
            f = x[:, :, :, :, i]

            f = self.conv1(f)
            f = self.bn1(f)
            f = self.relu(f)
            f = self.maxpool(f)

            # f = F.dropout(f, p=0.5, training=self.training)

            f = self.layer1(f)
            f = self.layer2(f)
            f = self.layer3(f)
            f = self.layer4(f)

            f = self.avgpool(f)
            f = f.squeeze(3).squeeze(2)

            vs.append(f)

            alphas.append(self.alpha(f))
        vs_stack = torch.stack(vs, dim=2)
        alphas_stack = torch.stack(alphas, dim=2)
        alphas_stack = F.softmax(alphas_stack, dim=2)

        alphas_part_max = alphas_stack[:, :, 0:5].max(dim=2)[0]

        alphas_org = alphas_stack[:, :, 5]
        vm = vs_stack.mul(alphas_stack).sum(2).div(alphas_stack.sum(2))
        # vm = self.bn2(vm)
        # pdb.set_trace()
        for i in range(len(vs)):
            vs2.append(torch.cat([vs[i], vm], dim=1))
            # ---------------add---------------
            # vs[i] = F.dropout(vs[i], p=0.5, training=self.training)
        vs_stack_4096 = torch.stack(vs2, dim=2)
        # pdb.set_trace()
        betas = []
        for index, v in enumerate(vs2):
            betas.append(self.beta(v))
            # betas.append(F.dropout(self.beta(v), p = 0.5, training = self.training))
        betas_stack = torch.stack(betas, dim=2)
        betas_stack = F.softmax(betas_stack, dim=2)

        # output = vs_stack_4096.mul(betas_stack).sum(2).div(betas_stack.sum(2))
        # index_image = torch.max((betas_stack*alphas_stack),1)
        # pdb.set_trace()
        vm = vs_stack_4096.mul(betas_stack * alphas_stack).sum(2).div((betas_stack * alphas_stack).sum(2))

        # max, index = torch.max(betas_stack*alphas_stack)
        # pdb.set_trace()
        # output = F.dropout(output, p=0.5, training=self.training)
        # vm = self.bn3(vm)
        for i in range(len(vs)):
            vs2[i] = torch.cat([vs[i], vm], dim=1)
            # ---------------add---------------
            # vs[i] = F.dropout(vs[i], p=0.5, training=self.training)
        vs_stack_4096 = torch.stack(vs2, dim=2)
        # pdb.set_trace()
        deltas = []
        for index, v in enumerate(vs2):
            deltas.append(self.delta(v))
            # betas.append(F.dropout(self.beta(v), p = 0.5, training = self.training))
        deltas_stack = torch.stack(deltas, dim=2)
        deltas_stack = F.softmax(deltas_stack, dim=2)

        output = vs_stack_4096.mul(deltas_stack * betas_stack * alphas_stack).sum(2).div(
            (deltas_stack * betas_stack * alphas_stack).sum(2))

        output = output.view(output.size(0), -1)
        # output = self.bn4(output)
        pred_score = self.fc(output)

        self.output = output
        # pred_score = F.dropout(pred_score, p=0.2, training=self.training)
        # x = x.view(x.size(0), -1)

        return pred_score, alphas_part_max, alphas_org, alphas_stack[:, :, 0:5]


if __name__ == "__main__":
    # FAN
    net = FAN()
    input = torch.randn(1, 3, 128, 128)

    # MSAN
    net = MSAN(BasicBlock, [2, 2, 2, 2])
    input = torch.randn(1, 3, 128, 128, 6)
    output = net(input)
    print(output)