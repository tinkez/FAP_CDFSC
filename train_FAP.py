import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random

from methods.backbone import model_dict
from data.datamgr import SetDataManager
from methods.matchingnet import MatchingNet
from methods.relationnet import RelationNet
from methods.protonet import ProtoNet
from methods.gnnnet import GnnNet
from methods.tpn import TPN
from pytorch_wavelets import DWTForward, DWTInverse
from options import parse_args
from functions import *


def FAPLoop(X_n, X_n_low_randn, X_n_low_zeros, params, model):
    X_n = X_n.cuda()
    #Iden = X_n
    N, S, C, H, W = X_n.size()
    p = np.random.rand()
    K = [1, 3, 5, 7, 11, 15]
    if p > params.prob:
        k = K[np.random.randint(0, len(K))]
        RandConv = nn.Conv2d(3, 3, kernel_size=k, stride=1, padding=k//2, bias=False).cuda()
        nn.init.xavier_normal_(RandConv.weight)
        X_n = RandConv(X_n.reshape(-1, C, H, W)).reshape(N, S, C, H, W)
        X_n_low_randn = RandConv(X_n_low_randn.reshape(-1, C, H, W)).reshape(N, S, C, H, W)
        X_n_low_zeros = RandConv(X_n_low_zeros.reshape(-1, C, H, W)).reshape(N, S, C, H, W)
        X_n_low_randn = mutual_attention(X_n, X_n_low_randn) + mutual_attention(X_n_low_randn, X_n) + X_n_low_randn
        X_n_low_zeros = mutual_attention(X_n, X_n_low_zeros) + mutual_attention(X_n_low_zeros, X_n) + X_n_low_zeros
        X_n_hat = Grad_ascending(model, X_n.detach())
        X_n_randn_hat = mutual_attention(X_n_hat, X_n_low_randn) + mutual_attention(X_n_low_randn, X_n_hat) + X_n_low_randn
        X_n_zeros_hat = mutual_attention(X_n_hat, X_n_low_zeros) + mutual_attention(X_n_low_zeros, X_n_hat) + X_n_low_zeros

    else:
        X_n_hat = Grad_ascending(model, X_n.detach())
        X_n_randn_hat = Grad_ascending(model, X_n_low_randn.detach())
        X_n_zeros_hat = Grad_ascending(model, X_n_low_zeros.detach())
        X_n_randn_hat = mutual_attention(X_n_hat, X_n_randn_hat) + mutual_attention(X_n_randn_hat, X_n_hat) + X_n_randn_hat
        X_n_zeros_hat = mutual_attention(X_n_hat, X_n_zeros_hat) + mutual_attention(X_n_zeros_hat, X_n_hat) + X_n_zeros_hat
    return X_n_hat.detach(), X_n_randn_hat.detach(), X_n_zeros_hat.detach()

def mutual_attention(q, k):
  assert (q.size() == k.size())
  weight = q.mul(k)
  weight_sig = torch.sigmoid(weight)
  v = k.mul(weight_sig)
  return v

def Grad_ascending(model, X_n):
    X_n = X_n.cuda()
    optimizer = optim.SGD([X_n.requires_grad_()], lr=params.max_lr)
    model.eval()
    for _ in range(params.T_max):
        optimizer.zero_grad()
        _, class_loss = model.set_forward_loss(X_n)
        (-class_loss).backward()
        optimizer.step()
    return X_n.detach()

def train(base_loader, val_loader, model, start_epoch, stop_epoch, params):
    max_acc = 0.
    optimizer = torch.optim.Adam(model.parameters())
    print_freq = len(base_loader)//10
    for epoch in range(start_epoch, stop_epoch):
        avg_loss = 0.
        for i, (x, _) in enumerate(base_loader):
            x_low_rand, x_low_zeros = freq_trans_low(x)
            x_hat, x_randn_hat, x_zeros_hat = FAPLoop(x, x_low_rand, x_low_zeros, params, model)
            model.train()
            optimizer.zero_grad()
            scores_original, loss_original = model.set_forward_loss(x_hat)
            scores_randn, loss_randn = model.set_forward_loss(x_randn_hat)
            scores_zeros, loss_zeros = model.set_forward_loss(x_zeros_hat)
            kl_loss_randn = F.kl_div(scores_randn.softmax(dim=-1).log(), scores_original.softmax(dim=-1), reduction='batchmean')
            kl_loss_zeros = F.kl_div(scores_zeros.softmax(dim=-1).log(), scores_original.softmax(dim=-1), reduction='batchmean')
            loss = loss_original + loss_randn + loss_zeros + kl_loss_randn + kl_loss_zeros
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()
            if (i + 1) % print_freq == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i+1, len(base_loader), avg_loss/float(i+1)))
        model.eval()
        with torch.no_grad():
            acc = model.test_loop(val_loader)

        if acc > max_acc:
            print("Best model! save...")
            max_acc = acc
            outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
            torch.save({'epoch':epoch, 'state':model.state_dict()}, outfile)
        else:
            print("GG! Best accuracy {:f}".format(max_acc))

        if ((epoch+1) % params.save_freq == 0) or (epoch == stop_epoch-1):
            outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)
    return model

if __name__=='__main__':
    # set seed
    np.random.seed(0)
    # set random seed
    # seed = 10
    # print("set seed = %d" % seed)
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # parser argument
    params = parse_args()
    print('--- Training ---\n')
    print(params)

    # output dir
    params.checkpoint_dir = '%s/checkpoints/%s'%(params.save_dir, params.name)
    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    # dataloader
    print('\n--- Prepare dataloader ---')
    print('\ttrain with single seen domain {}'.format(params.dataset))
    print('\tval with single seen domain {}'.format(params.testset))
    base_file = os.path.join(params.data_dir, params.dataset, 'base.json')
    val_file = os.path.join(params.data_dir, params.testset, 'val.json')

    # model
    image_size = 224
    n_query = max(1, int(16*params.test_n_way/params.train_n_way))
    base_datamgr = SetDataManager(image_size, n_query=n_query, n_way=params.train_n_way, n_support=params.n_shot)
    base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)
    val_datamgr = SetDataManager(image_size, n_query=n_query, n_way=params.test_n_way, n_support=params.n_shot)
    val_loader = val_datamgr.get_data_loader(val_file, aug=False)

    if params.method == 'MatchingNet':
        model = MatchingNet(model_dict[params.model], n_way=params.train_n_way, n_support=params.n_shot).cuda()
    elif params.method == 'RelationNet':
        model = RelationNet(model_dict[params.model], n_way=params.train_n_way, n_support=params.n_shot).cuda()
    elif params.method == 'ProtoNet':
        model = ProtoNet(model_dict[params.model], n_way=params.train_n_way, n_support=params.n_shot).cuda()
    elif params.method == 'GNN':
        model = GnnNet(model_dict[params.model], n_way=params.train_n_way, n_support=params.n_shot).cuda()
    elif params.method == 'TPN':
        model = TPN(model_dict[params.model], n_way=params.train_n_way, n_support=params.n_shot).cuda()
    else:
        print("Please specify the method!")
        assert(False)
    model.n_query = n_query

    # load model
    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch
    if params.resume_epoch > 0:
        resume_file = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(params.resume_epoch))
        tmp = torch.load(resume_file)
        start_epoch = tmp['epoch']+1
        model.load_state_dict(tmp['state'])
        print('\tResume the training weight at {} epoch.'.format(start_epoch))
    else:
        path = '%s/checkpoints/%s/399.tar' % (params.save_dir, params.resume_dir)
        state = torch.load(path)['state']
        model_params = model.state_dict()
        pretrained_dict = {k: v for k, v in state.items() if k in model_params}
        print(pretrained_dict.keys())
        model_params.update(pretrained_dict)
        model.load_state_dict(model_params)

    # training
    print('\n--- start the training ---')
    model = train(base_loader, val_loader, model, start_epoch, stop_epoch, params)