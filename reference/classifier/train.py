# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
import argparse
import time
from datetime import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
# os.environ["WANDB_API_KEY"] = "968bf03e603c6d75abd722195aa954528b73424e"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from conf import settings
from utils import get_network, loaddata, loaddata2, loaddata_edge, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights

from sklearn.metrics import *
from torch.distributions import Categorical, kl_divergence

import models


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'  # 这里你可以替换成主节点的 IP
    os.environ['MASTER_PORT'] = '12356'      # 选择一个合适的端口
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    # 初始化进程组
    dist.init_process_group(
        backend="nccl",  # 使用 NCCL 后端（适用于 GPU）
        init_method="env://",  # 从环境变量中获取初始化信息
        rank=rank,
        world_size=world_size
    )

def cleanup():
    # 销毁进程组
    dist.destroy_process_group()


def train(epoch, args, net, ju_dataloader, optimizer, loss_function, rank, writer=None, warmup_scheduler=None):
    start = time.time()
    total_loss = 0
    net.train()
    for batch_index, (rgb_images, labels) in enumerate(ju_dataloader['train']):
        rgb_images = rgb_images.to(rank, non_blocking=True)
        labels = labels.to(rank, non_blocking=True)

        optimizer.zero_grad()
        outputs = net(rgb_images)
        loss = loss_function(outputs, labels).mean()
        # for name, para in net.named_parameters():
        #     print(f"Parameter: {name}, Requires Grad: {para.requires_grad}, Grad is None: {para.grad is None}")
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_iter = (epoch - 1) * len(ju_dataloader) + batch_index + 1


        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(rgb_images),
            total_samples=len(ju_dataloader)
        ))

    avg_loss = total_loss / len(ju_dataloader['train'])

    if epoch <= args.warm:
        warmup_scheduler.step()

    wandb.log({"train_loss": avg_loss})

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
# def eval_training(epoch=0, tb=True):
def eval_training(epoch, net, test_loader, loss_function, rank, world_size, writer=None):
    predict_labels = []
    true_labels = []
    start = time.time()
    
    # 设置网络为评估模式
    net.eval()

    test_loss = 0.0  # 总损失
    correct = 0.0    # 总正确预测数
    class_correct = {}  # 存储每个类别的正确预测数量
    class_total = {}    # 存储每个类别的总数量

    # 遍历测试数据
    for (rgb_images, labels) in test_loader['test']:
        rgb_images = rgb_images.to(rank, non_blocking=True)
        labels = labels.to(rank, non_blocking=True)

        # 前向传播
        outputs = net(rgb_images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()

        # 计算预测结果
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

        predict_labels.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

        # 统计每个类别的正确预测数量和总样本数量
        for label, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
            if label not in class_correct:
                class_correct[label] = 0
                class_total[label] = 0
            if label == pred:
                class_correct[label] += 1
            class_total[label] += 1

    # # 收集各个进程上的统计信息（多卡环境）
    # total_loss = torch.tensor(test_loss).to(rank)  
    # total_correct = correct.clone().detach().to(rank)  # 使用 clone().detach()
    
    # dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    # dist.all_reduce(total_correct, op=dist.ReduceOp.SUM)
    
    # 将预测和标签转换为 numpy 数组
    predict_labels = np.array(predict_labels)
    true_labels = np.array(true_labels)

    # 每个类别的精度、召回率、F1分数计算
    if rank == 0:
        print("Each class accuracy:")
        for label in sorted(class_total.keys()):
            accuracy = 100 * class_correct[label] / class_total[label]
            precision = 100 * precision_score(true_labels, predict_labels, labels=[label], average='macro', zero_division=0)
            recall = 100 * recall_score(true_labels, predict_labels, labels=[label], average='macro', zero_division=0)
            f1 = 100 * f1_score(true_labels, predict_labels, labels=[label], average='macro', zero_division=0)

            print(f"类别 {label}:")
            print(f"准确率: {accuracy:.2f}%")
            print(f"精确率：{precision:.2f}%")
            print(f"召回率: {recall:.2f}%")
            print(f"F1-Score: {f1:.2f}%")
            print()

        # 计算整体准确率
        accuracy = accuracy_score(true_labels, predict_labels)
        print("整体精度: ", accuracy)
        
        avg_loss = test_loss / len(test_loader['test'])
        
        # 精度、查准率、召回率、F1-Score的宏观平均
        precision_macro = precision_score(true_labels, predict_labels, average='macro')
        recall_macro = recall_score(true_labels, predict_labels, average='macro')
        f1_macro = f1_score(true_labels, predict_labels, average='macro')

        print(f"宏观平均：")
        print(f"查准率P: {precision_macro:.2f}")
        print(f"召回率R: {recall_macro:.2f}")
        print(f"F1-Score: {f1_macro:.2f}")

    # 评估结束时间
    finish = time.time()

    print('Evaluating Network.....')
    # print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed: {:.2f}s'.format(
    #     epoch, 
    #     total_loss.item() / len(test_loader['test']),
    #     total_correct.float() / len(test_loader['test']),
    #     finish - start
    # ))

    wandb.log({"test_loss": avg_loss})
    wandb.log({"test_accuracy": accuracy})
    wandb.log({"F1-score": f1_macro})
    return accuracy

def main(rank, world_size, args):
    # rank = 0  # Replace with actual rank
    # world_size = 1  # Replace with actual world size

    # 初始化分布式环境
    setup(rank, world_size)
    # 设置当前 GPU
    torch.cuda.set_device(rank)
    # 初始化TensorBoard（仅在主进程）
    writer = SummaryWriter(log_dir=os.path.join(settings.LOG_DIR, args.net, settings.TIME_NOW)) if rank == 0 else None
    # 加载模型
    net = get_network(args)
    net = net.to(rank)
    # 使用 DDP 包装模型
    net = DDP(net, device_ids=[rank], find_unused_parameters=True)

    loss_function = nn.CrossEntropyLoss()
    # loss_function = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-1)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(loaddata(args.data, batch_size=16, set_name='train', shuffle=True, num_workers=args.num_workers)['train'])
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    #dataset
    # 加载数据
    train_loader = loaddata(args.data, batch_size=args.b, set_name='train', shuffle=False, num_workers=args.num_workers, rank=rank, world_size=world_size)
    test_loader = loaddata(args.data, batch_size=args.b, set_name='test', shuffle=False, num_workers=args.num_workers, rank=rank, world_size=world_size)
    # # 使用 DistributedSampler
    # train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    # test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    # train_loader = DataLoader(train_dataset, batch_size=args.b, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    # test_loader = DataLoader(test_dataset, batch_size=args.b, sampler=test_sampler, num_workers=args.num_workers, pin_memory=True)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_file_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))
        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))

    wandb.init(project="ELPV-resnet50-v1")

    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        # train(epoch, args)
        train(epoch, args, net, train_loader, optimizer, loss_function, rank, writer, warmup_scheduler)
        # acc = eval_training(epoch)
        if rank == 0 and epoch % 5 == 0:  # 只在主进程上验证
            acc = eval_training(epoch, net, test_loader, loss_function, rank, world_size=world_size, writer=writer)

        #start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_file_path.format(net=args.net, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_file_path.format(net=args.net, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

    if writer is not None and rank == 0:
        writer.close()
    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='DGCF_Resnet18', help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-num_workers', type=int, default=8)
    parser.add_argument('-b', type=int, default=12, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=5, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=5e-4, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-data', type=str, default='/disk2/zhuy/PV/EL/masked_EL/baseline_dataset/')
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
