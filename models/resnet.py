import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch


class ResNet(nn.Module):
    def __init__(self, base_model_name: str, pretrained=False,
                 num_classes=264):
        super().__init__()
        base_model = models.__getattribute__(base_model_name)(
            pretrained=pretrained)
        layers = list(base_model.children())[:-2]
        layers.append(nn.AdaptiveMaxPool2d(1))
        self.encoder = nn.Sequential(*layers)

        in_features = base_model.fc.in_features
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(1024, num_classes))

    def forward(self, x):
        batch_size = x.size(0)
        x = self.encoder(x)
        x = x.view(batch_size, -1)
        x = self.classifier(x)
        multiclass_proba = F.softmax(x, dim=1)
        multilabel_proba = torch.sigmoid(x)
        # return {
        #     "logits": x,
        #     "multiclass_proba": multiclass_proba,
        #     "multilabel_proba": multilabel_proba
        # }
        return x

"""
# https://zhuanlan.zhihu.com/p/93806755
class res50(torch.nn.Module):
    def __init__(self, num_classes):
        super(res50, self).__init__()
        resnet = resnet50(pretrained=True)
        self.backbone = torch.nn.Sequential(
                        resnet.conv1,
                        resnet.bn1,
                        resnet.relu,
                        resnet.layer1,
                        resnet.layer2,
                        resnet.layer3,
                        resnet.layer4
        )
        self.pool = torch.nn.AdaptiveMaxPool2d(1)
        self.bnneck = nn.BatchNorm1d(2048)
        self.bnneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(2048, num_classes, bias=False)
    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        feat = x.view(x.shape[0], -1)
        feat = self.bnneck(feat)
        if not self.training:
            return nn.functional.normalize(feat, dim=1, p=2)
        x = self.classifier(feat)
        return x
"""