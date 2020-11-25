import os
import sys
import argparse
import time
from tqdm import tqdm, notebook

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

from utils import AverageMeter, SupConLoss, CosineScheduler, NormLayer, TwinAug, save, load, top_n_acc, set_args

_CIFAR_MEAN, _CIFAR_STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

def pretrain(args, model, optimizer, Loss, dataloader, scheduler=None, ckpt=0):
    model.train()
    
    # record losses over epochs
    loss_total = AverageMeter(name='Pretrain Loss', save_all=True, save_dir='./results/{}'.format(args.name))
    
    for e in range(ckpt + 1, args.pretrain_epochs + 1):
        loss_epoch = AverageMeter()
        if scheduler is not None:
            scheduler.step()
        epoch_iterator = tqdm(dataloader, desc="Pretrain Iteration {}".format(e))
        for it, batch in enumerate(epoch_iterator):
            optimizer.zero_grad()
            inputs, labels = batch[0].cuda(), batch[1].cuda()
            d = inputs.size()
            inputs = inputs.view(2 * d[0], d[2], d[3], d[4])
            reps = model(inputs)
            loss = Loss(reps, labels.unsqueeze(1).expand(d[0], 2).reshape(-1))
            loss.backward()
            loss_epoch.update(loss.item(), len(labels))
            optimizer.step()
        print("[Pretrain {}] loss: {}".format(e, loss_epoch.avg))
        loss_total.update(loss_epoch.avg)
        
        # save model
        if (e % args.pretrain_save_period) == 0:
            save(args, "pretrain_e{}".format(e), {"model": model.state_dict(), "optimizer": optimizer.state_dict()})
            loss_total.save()
        
            
def finetune(args, model, optimizer, Loss, dataloader, eval_loader, scheduler=None, ckpt=0):
    model.train()
    
    # manage perfomances of model and losses over epochs
    eval_state = {"size": 0, "top1": AverageMeter(name='Top1-Acc (%)', save_all=True, save_dir='./results/{}'.format(args.name)), "top5": AverageMeter(name='Top5-Acc (%)', save_all=True, save_dir='./results/{}'.format(args.name))}
    loss_total = AverageMeter(name='Finetune Loss', save_all=True, save_dir='./results/{}'.format(args.name))
    
    for e in range(ckpt + 1, args.finetune_epochs + 1):
        loss_epoch = AverageMeter()
        if scheduler is not None:
            scheduler.step()
        epoch_iterator = tqdm(dataloader, desc="Finetune Iteration {}".format(e))
        for it, batch in enumerate(epoch_iterator):
            optimizer.zero_grad()
            inputs, labels = batch[0].cuda(), batch[1].cuda()
            logits = model(inputs)
            loss = Loss(logits, labels)
            loss.backward()
            loss_epoch.update(loss.item())
            optimizer.step()
        print("[Finetune {}] loss: {}".format(e, loss_epoch.sum))
        loss_total.update(loss_epoch.sum)
        
        # evaluate model
        if (e % args.eval_period) == 0:
            eval(args, model, eval_loader, state=eval_state)
            model.train()
        
        # save model
        if (e % args.finetune_save_period) == 0:
            save(args, "finetune_e{}".format(e), {"model": model.state_dict(), "optimizer": optimizer.state_dict()})
            eval_state["top1"].save()
            eval_state["top5"].save()
            loss_total.save()

def eval(args, model, dataloader, state=None):
    model.eval()
    top1_acc = AverageMeter()
    top5_acc = AverageMeter()
    eval_iterator = tqdm(dataloader, desc="Test Iteration")
    for it, batch in enumerate(eval_iterator):
        inputs, labels = batch[0].cuda(), batch[1].cuda()
        logits = model(inputs)
        preds = nn.Softmax(dim=1)(logits)
        top1 = top_n_acc (preds, labels, n=1)
        top5 = top_n_acc (preds, labels, n=5)
        top1_acc.update(top1, len(labels))
        top5_acc.update(top5, len(labels))
    state['size'] += 1
    top1_total = top1_acc.avg * 100
    top5_total = top5_acc.avg * 100
    print("[Test {}] top1: {}%, top5: {}%".format(state['size'], top1_total, top5_total))
    is_best, _ = state["top1"].update(top1_total)
    state["top5"].update(top5_total)
    
    # save best model
    if is_best:
        save(args, "best", model.state_dict())
        

def main():
    args = set_args()
    
    # 0. initial setting
    print("[{}] starts".format(args.name))
    # set environmet
    if not os.path.isdir(os.path.join('./ckpt', args.name)):
        os.mkdir(os.path.join('./ckpt', args.name))
    if not os.path.isdir(os.path.join('./results', args.name)):
        os.mkdir(os.path.join('./results', args.name))
    cudnn.bechmark = True
    # 1. load data  
    print('Loading data...')
    
    # augmentation
    aug_pretrain = transforms.Compose([transforms.RandomCrop(28), transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.5, 0.5, 0.5, 0.5), transforms.ToTensor(), transforms.Normalize(mean=_CIFAR_MEAN, std=_CIFAR_STD)])
    aug_finetune = transforms.Compose([transforms.RandomCrop(28), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean=_CIFAR_MEAN, std=_CIFAR_STD)])
    aug_test = transforms.Compose([transforms.RandomCrop(28), transforms.ToTensor(), transforms.Normalize(mean=_CIFAR_MEAN, std=_CIFAR_STD)])
    
    # dataset
    if args.use_cifar100:
        print("use cifar100")
        pretrain_dataset = datasets.CIFAR100('./data/', train=True, download=True, transform=TwinAug(aug_pretrain, aug_pretrain))
        finetune_dataset = datasets.CIFAR100('./data/', train=True, download=True, transform=aug_finetune)
        test_dataset = datasets.CIFAR100('.data/', train=True, download=True, transform=aug_test)
    else:
        print("use cifar10")
        pretrain_dataset = datasets.CIFAR10('./data/', train=True, download=True, transform=TwinAug(aug_pretrain, aug_pretrain))
        finetune_dataset = datasets.CIFAR10('./data/', train=True, download=True, transform=aug_finetune)
        test_dataset = datasets.CIFAR10('.data/', train=True, download=True, transform=aug_test)
    
    # dataloader
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    finetune_loader = DataLoader(finetune_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 2. model
    print('Setting model...')
    model = nn.DataParallel(models.resnet50())
    model.cuda()
    embedding_size = model.module.fc.in_features

    # 3. Loss
    print('Setting loss...')
    pretrain_loss = SupConLoss(temperature=args.temperature)
    finetune_loss = nn.CrossEntropyLoss()
    
    # 4. pretrain
    # attach projection layer
    model.module.fc = nn.Sequential(NormLayer(), nn.Linear(embedding_size, 2048), nn.ReLU(), nn.Linear(2048, 128), NormLayer())
    model.cuda()
    
    pretrain_optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, nesterov=True)
    pretrain_scheduler = CosineScheduler(pretrain_optimizer, lr=args.lr, warmup=args.warmup, epochs=args.pretrain_epochs)
    pretrain_ckpt = 0
    
    # resume from checkpoint
    if args.pretrain_ckpt is not None:
        states = load(args, args.pretrain_ckpt, map_location='cuda:0')
        model.load_state_dict(states['model'])
        pretrain_optimizer.load_state_dict(states['optimizer'])
        pretrain_ckpt = 200 #states['epoch']
        
    if not args.pass_pretrain:
        print('Pretraining...')
        pretrain_scheduler.jump(pretrain_ckpt)
        pretrain(args, model, pretrain_optimizer, pretrain_loss, pretrain_loader, scheduler=pretrain_scheduler, ckpt=pretrain_ckpt)

    # 5. finetune
    # fix encoder and attach classification layer
    for params in model.parameters():
        params.requires_grad = False
    model.module.fc = nn.Sequential(NormLayer(), nn.Linear(embedding_size, 100 if args.use_cifar100 else 10))
    model.cuda()
 
    finetune_optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=0.0005, nesterov=True)
    finetune_ckpt = 0
    
    # resume from checkpointi
    if args.finetune_ckpt is not None:
        states = load(args, args.finetune_ckpt, map_location='cuda:0')
        model.load_state_dict(states['model'])
        finetune_optimizer.load_state_dict(states['optimizer'])
        finetune_ckpt = states['epoch']
        
    if not args.pass_finetune:
        print('Finetuning...')
        finetune(args, model, finetune_optimizer, finetune_loss, finetune_loader, test_loader, ckpt=finetune_ckpt)
    
if __name__ == "__main__":
    main()
    
    
    
    
    
    
    
    
            
            
            
    
            
            
        