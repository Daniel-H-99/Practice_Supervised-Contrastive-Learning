import math
import argparse
import os
import torch
from matplotlib import pyplot as plt

# data manager for recording, saving, and plotting
class AverageMeter(object):
    def __init__(self, name='noname', save_all=False, save_dir='.'):
        self.name = name
        self.save_all = save_all
        self.save_dir = save_dir
        self.reset()
    def reset(self):
        self.max = - 100000000
        self.min = 100000000
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        if self.save_all:
            self.data = []
    def update(self, val, weight=1):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count
        if self.save_all:
            self.data.append(val)
        is_max, is_min = False, False
        if val > self.max:
            self.max = val
            is_max = True
        if val < self.min:
            self.min = val
            is_min = True
        return (is_max, is_min)
    def save(self):
        with open(os.path.join(self.save_dir, "{}.txt".format(self.name)), "w") as file:
            file.write("max: {0:.4f}\nmin: {1:.4f}".format(self.max, self.min))
        if self.save_all:
            plot = plt.figure()
            plt.plot(range(1, len(self.data) + 1), self.data)
            plt.ylabel(self.name)
            plt.savefig("{}/{}.png".format(self.save_dir, self.name))
            plt.close(plot)

# Supervised Constrastive Loss
class SupConLoss(object):
    def __init__(self, temperature=1):
        self.temperature = temperature
    def __call__(self, features, labels):
        num = labels.size()[0]
        cluster_mat = labels == labels.unsqueeze(1).expand(-1, num)
        relation_mat = features @ features.t()
#         print(relation_mat)
        relation_mat = torch.exp(relation_mat / self.temperature)
        loss = -torch.log(torch.clamp(((relation_mat * cluster_mat).sum(1) - relation_mat.diag()) / (relation_mat.sum(1) - relation_mat.diag()), min=1e-8))
        loss = loss / (cluster_mat.sum(1) - 1)
        loss = loss.sum()
        return loss
    
# Consine Annealing Scheduler with linear warmup
class CosineScheduler(object):
    def __init__(self, optimizer, lr=0.1, warmup=0, epochs=200, ets=1e-8):
        self.optimizer = optimizer
        self.lr = lr
        self.warmup = warmup
        self.epochs = epochs
        self.cur = 0
        self.ets = ets
    def step(self):
        self.cur += 1
        if self.cur <= self.warmup:
            self.optimizer.lr = max(self.ets, self.lr * self.cur / self.warmup)
        else:
            self.optimizer.lr = max(self.ets, self.lr * math.cos(math.pi * self.cur / (2 * self.epochs)))
    def jump(self, epoch):
        self.cur = epoch

# pair of augmentation for constructin multi-view batch
class TwinAug(object):
    def __init__(self, tf1, tf2):
        self.tf1 = tf1
        self.tf2 = tf2
    def __call__(self, input):
        return torch.stack([self.tf1(input), self.tf2(input)])

# layer for normalizing features
class NormLayer(torch.nn.Module):
    def __init__(self):
        super(NormLayer, self).__init__()
    def forward(self, x):
        norm = torch.norm(x, p=2, dim=1).detach()
        x = x.div(norm.unsqueeze(1).expand_as(x))
        return x
    
# construct args
def set_args():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--name', type=str, default='SupCon', help="tag for experiment")
    parser.add_argument('--pass_pretrain', action='store_true', help="conduct pretraining")
    parser.add_argument('--pass_finetune', action='store_true', help="conduct finetuning")
    parser.add_argument('--pretrain_ckpt', type=str, default=None, help="checkpoint in pretraining")
    parser.add_argument('--finetune_ckpt', type=str, default=None, help="checkpoint in finetuning")
    parser.add_argument('--pretrain_epochs', type=int, default=350, help="rounds of pretraining")
    parser.add_argument('--pretrain_save_period', type=int, default=2, help="rounds of pretraining") 
    parser.add_argument('--finetune_epochs', type=int, default=50, help="rounds of pretraining")
    parser.add_argument('--finetune_save_period', type=int, default=1, help="rounds of pretraining")
    parser.add_argument('--eval_period', type=int, default=1, help="rounds of pretraining")
    parser.add_argument('--use_cifar100', action='store_true', help="rounds of pretraining")
    parser.add_argument('--warmup', type=int, default=10, help="rounds of pretraining")
    parser.add_argument('--batch_size', type=int, default=512, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt', help="directory for checkpoints")
    parser.add_argument('--temperature', type=float, default=0.1, help='temperature for softmax')
    
    args = parser.parse_args()
    return args

# save data, such as model, optimizer
def save(args, surfix, data):
    torch.save(data, os.path.join(args.ckpt_dir, args.name, "{}.pt".format(surfix)))

# load data, such as model, optimizer
def load(args, surfix, map_location='cpu'):
    return torch.load(os.path.join(args.ckpt_dir, "{}.pt".format(surfix)), map_location=map_location)
  
# calculate top n accuracy
def top_n_acc (preds, labels, n=1):
    num = labels.size()[0]
    top_n_preds = (-preds).argsort(dim=1)[:, :n]
    corrects = (top_n_preds == labels.unsqueeze(1).expand(-1, n)).sum()
    return corrects.item() / num

