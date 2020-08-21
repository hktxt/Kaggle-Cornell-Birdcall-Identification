import torch.nn as nn
from torchvision import models
import pretrainedmodels
from resnest.torch import resnest50, resnest101
from efficientnet_pytorch import EfficientNet
from models.pyconvhgresnet import pyconvhgresnet50
from models.resnet_sk2 import sk2_resnet50
from models.resnet import ResNet


def get_model(model_name, classes=264, pretrained=True):
    print(f'loading {model_name}')

    if "resnet" in model_name:
        model = ResNet(  # type: ignore
            base_model_name=model_name,
            pretrained=pretrained,
            num_classes=classes)
        return model
    elif "resnest50" in model_name:
        model = get_resnest("resnest50", classes=classes, pretrained=pretrained)
        return model
    elif "efficientnet" in model_name:
        model = get_effdet(model_name, classes=classes)
        return model
    elif "pyconvhgresnet" in model_name:
        model = get_pyconvhgresnet(classes=classes, pretrained=pretrained)
        return model
    elif "resnet_sk2" in model_name:
        model = get_resnet_sk2(classes=classes, pretrained=pretrained)
        return model
    elif "se_resnet50_32x4d" in model_name:
        model = get_se_resnet50_32x4d(classes=classes, pretrained=pretrained)
        return model
    else:
        raise NotImplementedError


def get_resnest(net='resnest50', classes=264, pretrained=True):
    if net == 'resnest50':
        model = resnest50(pretrained=pretrained)
        del model.fc
        # # use the same head as the baseline notebook.
        model.fc = nn.Sequential(
            nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(1024, classes))
        return model


def get_effdet(modelName, classes, pretrained=True):
    model = EfficientNet.from_pretrained(modelName)
    del model._fc

    if modelName == "efficientnet-b0" or modelName == "efficientnet-b1":
        in_channels = 1280
    elif modelName == "efficientnet-b2":
        in_channels = 1408
    elif modelName == "efficientnet-b3":
        in_channels = 1536
    else:
        raise ValueError

    model._fc = nn.Sequential(
        nn.Linear(in_channels, 1024), nn.ReLU(), nn.Dropout(p=0.2),
        nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.2),
        nn.Linear(1024, classes))

    return model


def get_pyconvhgresnet(classes=264, pretrained=True):
    model = pyconvhgresnet50(pretrained=pretrained, num_classes=classes)
    return model


def get_resnet_sk2(classes=264, pretrained=True):
    model = sk2_resnet50(pretrained=pretrained, num_classes=classes)
    return model


def get_se_resnet50_32x4d(classes=264, pretrained=True):
    model = pretrainedmodels.__dict__['se_resnext50_32x4d'](num_classes=classes, pretrained='imagenet')
    return model