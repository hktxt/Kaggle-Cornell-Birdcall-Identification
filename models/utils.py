import torch.nn as nn
import torch
import math
from torch.autograd import Variable


class AngleLinear(nn.Module):  # 定义最后一层
    def __init__(self, in_features, out_features, m=3, phiflag=True):  # 输入特征维度，输出特征维度，margin超参数
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))  # 本层权重
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)  # 初始化权重，在第一维度上做normalize
        self.m = m
        self.phiflag = phiflag
        self.mlambda = [
            lambda x: x**0,
            lambda x: x**1,
            lambda x: 2*x**2-1,
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]  # 匿名函数,用于得到cos_m_theta

    @staticmethod
    def myphi(x, m):
        x = x * m
        return 1 - x ** 2 / math.factorial(2) + x ** 4 / math.factorial(4) - x ** 6 / math.factorial(6) +\
               x ** 8 / math.factorial(8) - x ** 9 / math.factorial(9)

    def forward(self, x):  # 前向过程，输入x
        w = self.weight

        ww = w.renorm(2, 1, 1e-5).mul(1e5)  # 方向0上做normalize
        x_len = x.pow(2).sum(1).pow(0.5)
        w_len = ww.pow(2).sum(0).pow(0.5)

        cos_theta = x.mm(ww)
        cos_theta = cos_theta / x_len.view(-1, 1) / w_len.view(1, -1)
        cos_theta = cos_theta.clamp(-1, 1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)  # 由m和/cos(/theta)得到cos_m_theta
            theta = Variable(cos_theta.data.acos())
            k = (self.m*theta/3.14159265).floor()
            n_one = k*0.0 - 1
            phi_theta = (n_one**k) * cos_m_theta - 2*k  # 得到/phi(/theta)
        else:
            theta = cos_theta.acos()  # acos得到/theta
            phi_theta = self.myphi(theta, self.m)  # 得到/phi(/theta)
            phi_theta = phi_theta.clamp(-1*self.m, 1)  # 控制在-m和1之间

        cos_theta = cos_theta * x_len.view(-1, 1)
        phi_theta = phi_theta * x_len.view(-1, 1)
        output = [cos_theta, phi_theta]  # 返回/cos(/theta)和/phi(/theta)
        return output
