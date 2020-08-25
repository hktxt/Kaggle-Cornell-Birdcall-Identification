import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F


class AngleLoss(nn.Module):  # 设置loss，超参数gamma，最小比例，和最大比例
    def __init__(self, gamma=0, lambda_min=5, lambda_max=1500):
        super(AngleLoss, self).__init__()
        self.gamma = gamma
        self.it = 0
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

    def forward(self, x, y):  # 分别是output和target
        self.it += 1
        cos_theta, phi_theta = x  # output包括上面的[cos_theta, phi_theta]
        y = y.view(-1, 1)

        index = cos_theta.data * 0.0
        index.scatter_(1, y.data.view(-1, 1), 1)  # 将label存成稀疏矩阵
        index = index.byte()
        # index = Variable(index)   # warning occurs, change to following line. see link blew:
        # https://github.com/pytorch/pytorch/issues/29365
        #index = torch.tensor(index, dtype=torch.bool)
        index = index.clone().detach().bool()

        lamb = max(self.lambda_min, self.lambda_max / (1 + 0.1 * self.it))  # 动态调整lambda，来调整cos(\theta)和\phi(\theta)的比例
        output = cos_theta * 1.0
        output[index] -= cos_theta[index]*(1.0+0)/(1 + lamb)  # 减去目标\cos(\theta)的部分
        output[index] += phi_theta[index]*(1.0+0)/(1 + lamb)  # 加上目标\phi(\theta)的部分

        logpt = F.log_softmax(output, dim=1)
        logpt = logpt.gather(1, y)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()

        return loss


class AngleLossWithCE(nn.Module):
    def __init__(self, lambda_min=5, lambda_max=1500, weight=[1, 1]):
        super().__init__()
        self.embeddingLoss = AngleLoss(lambda_min, lambda_max)
        self.clsLoss = nn.CrossEntropyLoss()
        self.weight = weight

    def forward(self, x1, x2, label):
        embeddingLoss = self.embeddingLoss(x1, label)
        clsLoss = self.clsLoss(x2, label)
        total_loss = embeddingLoss * self.weight[0] + clsLoss * self.weight[1]
        return total_loss


# http://papers.nips.cc/paper/6653-learning-with-average-top-k-loss.pdf
# not sure working or not
class TopKLossWithBCE(nn.Module):
    def __init__(self, p=0.7):
        super().__init__()
        self.p = p
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred, gt):
        k = int(pred.shape[0] * self.p)
        loss = self.bce(pred, gt)
        loss = loss.topk(k, dim=0)[0]
        loss = loss.mean()
        return loss


# https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
class F1_Loss(nn.Module):
    '''Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''

    def __init__(self, classes=2, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
        self.classes = classes

    def forward(self, y_pred, y_true, ):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()


class F1LossWithBCE(nn.Module):
    def __init__(self, classes=264, weights=[1, 1]):
        super().__init__()
        self.classes = classes
        self.weights = weights
        self.bce = nn.BCEWithLogitsLoss()
        self.f1 = F1_Loss(classes=self.classes)

    def forward(self, pred, gt):
        bce = self.bce(pred, gt)
        f1 = self.f1(pred, gt)
        loss = self.weights[0] * bce + self.weights[1] * f1
        return loss


