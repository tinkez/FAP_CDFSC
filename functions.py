import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse
from options import parse_args

def one_hot(y, num_class):         
    return torch.zeros((len(y), num_class)).scatter_(1, y.unsqueeze(1), 1)

def freq_trans_low(x):
    X_low_randn = x.clone().cuda()
    X_low_zeros = x.clone().cuda()
    DWT_Forward = DWTForward(J=1, wave='haar', mode='zero').cuda()
    DWT_Inverse = DWTInverse(wave='haar', mode='zero').cuda()
    assert X_low_randn.equal(X_low_zeros)
    for i in range(X_low_randn.size(0)):
      Yl, Yh = DWT_Forward(X_low_randn[i])
      Yh_hat_randn = []
      Yh_hat_zeros = []
      for j in range(len(Yh)):
        # print('Yh[j].size:', Yh[j].size())
        Yh_hat_randn.append(torch.randn_like(Yh[j]))
        Yh_hat_zeros.append(torch.zeros_like(Yh[j]))
      Y_rec_low_randn = DWT_Inverse((Yl, Yh_hat_randn))
      Y_rec_low_zeros = DWT_Inverse((Yl, Yh_hat_zeros))
      X_low_randn[i] = Y_rec_low_randn
      X_low_zeros[i] = Y_rec_low_zeros
    return X_low_randn, X_low_zeros




