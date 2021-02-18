import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
import pretrainedmodels as pmodels
from efficientnet_pytorch import EfficientNet
# from multigrain.lib import get_multigrain


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth',
    'resnext101_32x16d': 'https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth',
    'resnext101_32x32d': 'https://download.pytorch.org/models/ig_resnext101_32x32-e4b90b00.pth',
    'resnext101_32x48d': 'https://download.pytorch.org/models/ig_resnext101_32x48-3e41cc8a.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    'densenet121':'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
    'inception_v3': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}

def resnet(num_classes=9, layers=101, state_dict=None):
    if layers == 18:
        model = models.resnet18()
    elif layers == 34:
        model = models.resnet34()
    elif layers == 50:
        model = models.resnet50()
    elif layers == 101:
        model = models.resnet101()
    elif layers == 152:
        model = models.resnet152()

    if state_dict is not None:
        print('load_state_dict')
        model.load_state_dict(state_dict)

    num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, num_classes)
    model.fc = nn.Sequential(
                    nn.Linear(num_ftrs, 256),
                    nn.ReLU(),
                    nn.BatchNorm1d(256),
                    nn.Dropout(0.5),
                    nn.Linear(256, num_classes))

    return model

def resnext(num_classes=9, layers=101, state_dict=None):
    if layers == 50:
        model = models.resnext50_32x4d()
    elif layers == 101:
        model = models.resnext101_32x8d()

    if state_dict is not None:
        model.load_state_dict(state_dict)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model

def resnext_wsl(num_classes=9, bottleneck_width=8):
    model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x'+str(bottleneck_width)+'d_wsl')

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def vgg_bn(num_classes=9, layers=16, state_dict=None):
    if layers == 16:
        model = models.vgg16_bn()
    elif layers == 19:
        model = models.vgg19_bn()

    if state_dict is not None:
        model.load_state_dict(state_dict)

    model._modules['6'] = nn.Linear(4096, num_classes)
    return model

def densenet(num_classes=9, layers=121, state_dict=None):
    '''
        layers: 121, 201, 161
    '''
    if layers == 121:
        model = models.densenet121()
    elif layers == 201:
        model = models.densenet201()
    elif layers == 161:
        model = models.densenet161()

    if state_dict is not None:
        model.load_state_dict(state_dict)

    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    return model

def inception_v3(num_classes=9, layers=101, state_dict=None):
    model = models.inception_v3()
    if state_dict is not None:
        model.load_state_dict(state_dict)

    aux_ftrs = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Linear(aux_ftrs, num_classes)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def dpn(num_classes=9, layers=92, pretrained=True):
    model = torch.hub.load('rwightman/pytorch-dpn-pretrained', 'dpn'+str(layers), pretrained=pretrained)

    in_chs = model.classifier.in_channels
    model.classifier = nn.Conv2d(in_chs, num_classes, kernel_size=1, bias=True)
    return model


class EffNet(nn.Module):
    def __init__(self, num_classes=9, layers=0, pretrained=False):
        super(EffNet, self).__init__()
        if pretrained:
            self.model = EfficientNet.from_pretrained('efficientnet-b'+str(layers))
        else:
            self.model = EfficientNet.from_name('efficientnet-b'+str(layers))
        num_ftrs = self.model._fc.in_features
        # self.bn = nn.BatchNorm1d(num_ftrs*2)
        # self.dropout = nn.Dropout(0.3)
        # self.fc1 = nn.Linear(num_ftrs*2, 256)
        # self.bn1 = nn.BatchNorm1d(256, affine=False)
        # self.dropout1 = nn.Dropout(0.3)
        # self.fc2 = nn.Linear(256, num_classes)

        self.model._fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        # features = self.model.extract_features(x)
        # f1 = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
        # f2 = F.adaptive_max_pool2d(features, 1).squeeze(-1).squeeze(-1)

        # x = torch.cat([f1, f2], 1)
        # x = self.dropout(self.bn(x))
        # # print(x.size())
        # x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        # x = self.fc2(x)
        # return x

        return self.model(x)

def effnet(num_classes=9, layers=0, pretrained=False):
    model = EffNet(num_classes, layers, pretrained)
    return model

# def pnasnet_m(num_classes=9, layers=5, pretrained=False):
#     model = get_multigrain(backbone='pnasnet5large', include_sampling=False, learn_p=True)
#     if pretrained:
#         model.load_state_dict(
#             torch.load('/home/lzw/.cache/torch/checkpoints/pnasnet5large-finetune500.pth')['model_state'])
#     num_ftrs = model.classifier.in_features
#     model.classifier = nn.Linear(num_ftrs, num_classes)
#     return model
















