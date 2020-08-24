import torch.nn as nn
from torchvision import models
import pretrainedmodels
from resnest.torch import resnest50, resnest101
from efficientnet_pytorch import EfficientNet
from models.pyconvhgresnet import pyconvhgresnet50
from models.resnet_sk2 import sk2_resnet50
from models.resnet import ResNet


def get_model(model_name, classes=264, vgg=False, pretrained=True):
    print(f'loading {model_name}')

    if model_name == "resnet50":
        model = ResNet(  # type: ignore
            base_model_name=model_name,
            pretrained=pretrained,
            num_classes=classes)
        return model
    elif "resnest50" in model_name:
        model = get_resnest("resnest50", classes=classes, vgg=vgg, pretrained=pretrained)
        return model
    elif "efficientnet" in model_name:
        model = get_effdet(model_name, classes=classes, vgg=vgg)
        return model
    elif "pyconvhgresnet" in model_name:
        model = get_pyconvhgresnet(classes=classes, vgg=vgg, pretrained=pretrained)
        return model
    elif "resnet_sk2" in model_name:
        model = get_resnet_sk2(classes=classes, vgg=vgg, pretrained=pretrained)
        return model
    elif "se_resnet50_32x4d" in model_name:
        model = get_se_resnet50_32x4d(classes=classes, vgg=vgg, pretrained=pretrained)
        return model
    else:
        raise NotImplementedError


def get_resnest(net='resnest50', classes=264, vgg=False, pretrained=True):
    if net == 'resnest50':
        model = resnest50(pretrained=pretrained)
        # https://arxiv.org/pdf/1806.05622.pdf
        # we apply N*1 Conv2D (support in the frequency domain) and 1*K avg_pool.
        # The benefit of this modification is that the network becomes invariant
        # to temporal position but not frequency, which is desirable for speech,
        # but not for images.
        # Bottleneck-281 [-1, 2048, 7, 18], feature map before avg_pool

        if vgg:
            del model.avgpool
            model.avgpool = nn.Sequential(
                nn.BatchNorm2d(2048),
                nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=(7, 1)),
                nn.AvgPool2d(kernel_size=(1, 18))
            )

        del model.fc
        model.fc = nn.Sequential(
            nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(1024, classes))
        return model


def get_effdet(modelName, classes, vgg=False, pretrained=True):
    model = EfficientNet.from_pretrained(modelName)

    if modelName == "efficientnet-b0" or modelName == "efficientnet-b1":
        in_channels = 1280
    elif modelName == "efficientnet-b2":
        in_channels = 1408
    elif modelName == "efficientnet-b3":
        in_channels = 1536
    else:
        raise ValueError

    # https://arxiv.org/pdf/1806.05622.pdf
    # we apply N*1 Conv2D (support in the frequency domain) and 1*K avg_pool.
    # The benefit of this modification is that the network becomes invariant
    # to temporal position but not frequency, which is desirable for speech,
    # but not for images.
    # MemoryEfficientSwish-276   [-1, 1280, 7, 17], feature map before avg_pool

    if vgg:
        del model._avg_pooling
        model._avg_pooling = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=(7, 1)),
            nn.AvgPool2d(kernel_size=(1, 17))
        )

    del model._fc
    model._fc = nn.Sequential(
        nn.Linear(in_channels, 1024), nn.ReLU(), nn.Dropout(p=0.2),
        nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.2),
        nn.Linear(1024, classes))

    return model


def get_pyconvhgresnet(classes=264, vgg=False, pretrained=True):
    model = pyconvhgresnet50(pretrained=pretrained)
    # https://arxiv.org/pdf/1806.05622.pdf
    # we apply N*1 Conv2D (support in the frequency domain) and 1*K avg_pool.
    # The benefit of this modification is that the network becomes invariant
    # to temporal position but not frequency, which is desirable for speech,
    # but not for images.
    # AdaptiveAvgPool2d-212  [-1, 2048, 7, 18], feature map before avgpool

    if vgg:
        del model.avgpool
        model.avgpool = nn.Sequential(
            nn.BatchNorm2d(2048),
            nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=(7, 1)),
            nn.AvgPool2d(kernel_size=(1, 18))
        )
    del model.fc
    model.fc = nn.Sequential(
        nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(p=0.2),
        nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.2),
        nn.Linear(1024, 264))
    return model


def get_resnet_sk2(classes=264, vgg=False, pretrained=True):
    model = sk2_resnet50(pretrained=pretrained)
    # https://arxiv.org/pdf/1806.05622.pdf
    # we apply N*1 Conv2D (support in the frequency domain) and 1*K avg_pool.
    # The benefit of this modification is that the network becomes invariant
    # to temporal position but not frequency, which is desirable for speech,
    # but not for images.
    # Bottleneck-300   [-1, 2048, 7, 18], feature map before avg_pool

    if vgg:
        del model.avgpool
        model.avgpool = nn.Sequential(
            nn.BatchNorm2d(2048),
            nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=(7, 1)),
            nn.AvgPool2d(kernel_size=(1, 18))
        )

    del model.fc
    model.fc = nn.Sequential(
        nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(p=0.2),
        nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.2),
        nn.Linear(1024, classes))
    return model


def get_se_resnet50_32x4d(classes=264, vgg=False, pretrained=True):
    model = pretrainedmodels.__dict__['se_resnext50_32x4d'](pretrained='imagenet')

    # https://arxiv.org/pdf/1806.05622.pdf
    # we apply N*1 Conv2D (support in the frequency domain) and 1*K avg_pool.
    # The benefit of this modification is that the network becomes invariant
    # to temporal position but not frequency, which is desirable for speech,
    # but not for images.
    # SEResNeXtBottleneck-268 [-1, 2048, 7, 18], feature map before avg_pool
    if vgg:
        del model.avg_pool
        model.avg_pool = nn.Sequential(
            nn.BatchNorm2d(2048),
            nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=(7, 1)),  # !
            nn.AvgPool2d(kernel_size=(1, 18))  # !
        )

    del model.last_linear
    model.last_linear = nn.Sequential(
        nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(p=0.2),
        nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.2),
        nn.Linear(1024, classes))

    return model