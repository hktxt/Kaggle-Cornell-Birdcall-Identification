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


