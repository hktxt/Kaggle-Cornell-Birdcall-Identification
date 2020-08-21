from resnest.torch import resnest50, resnest101
import torch.nn as nn


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


if __name__ == "__main__":
    model = get_resnest()
    print(model)