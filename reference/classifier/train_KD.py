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
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.resnet import resnet18, DGCF_Resnet, DGCF_Resnet18
from models.convnext import DGCF_ConvNeXt

from conf import settings
from utils import get_network, loaddata, loaddata_edge, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights

from sklearn.metrics import *
from torch.distributions import Categorical, kl_divergence


# if torch.cuda.is_available():
#     print(f"Available GPUs: {torch.cuda.device_count()}")
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")
#     print("CUDA is not available, using CPU.")

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

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.2, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # 类别权重（可设置为类别数量倒数或实验值）
        self.gamma = gamma  # 困难样本调节因子（常用2-5）
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # 正确类别的预测概率
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DynamicWeightedLoss(nn.Module):
    def __init__(self, T_init=10, T_end=5, total_epochs=100):
        super().__init__()
        self.soft_loss = nn.KLDivLoss(reduction='batchmean')  ##改！！
        self.hard_loss = nn.CrossEntropyLoss()
        self.T_init = T_init  # 初始温度参数
        self.T_end = T_end    # 最终温度参数
        self.total_epochs = total_epochs

    def conv_cosine_loss(self, teacher_feats, student_feats, 
                        spatial_aware=True, 
                        project_student=False,
                        temperature=0.07):
        """
        卷积特征专用余弦相似度损失
        Args:
            teacher_feats: [B, C_t, H, W] 教师特征
            student_feats: [B, C_s, H, W] 学生特征
            spatial_aware: 是否保留空间维度计算相似度
            project_student: 是否对student特征进行通道投影
            temperature: 对比学习中的温度系数
        Returns:
            余弦相似度损失
        """
        # 可选：通道维度投影（当C_t ≠ C_s时必需）
        if project_student:
            projector = nn.Conv2d(student_feats.shape[1], teacher_feats.shape[1], 
                                kernel_size=1).to(student_feats.device)
            student_feats = projector(student_feats)  # [B, C_t, H, W]

        # 标准化（沿通道维度）
        teacher_norm = F.normalize(teacher_feats, p=2, dim=1)  # [B,C,H,W]
        student_norm = F.normalize(student_feats, p=2, dim=1)

        # 空间感知模式
        if spatial_aware:
            # 计算每个空间位置的相似度
            similarity_map = torch.sum(teacher_norm * student_norm, dim=1)  # [B,H,W]
            loss = 1 - similarity_map.mean()
        
        # 全局模式（忽略空间位置）
        else:
            # 展平空间维度
            t_flat = teacher_norm.flatten(2)  # [B, C, H*W]
            s_flat = student_norm.flatten(2)
            
            # 计算每个通道向量的相似度
            similarity = torch.einsum('bcp,bcq->bpq', t_flat, s_flat)  # [B, H*W, H*W]
            loss = 1 - similarity.mean() / temperature  # 可选温度系数
        return loss

    def forward(self, student_logits, ps1, teacher_logits, labels, current_epoch):
        # # 动态计算温度参数（余弦退火）
        # T = self.T_end + 0.5 * (self.T_init - self.T_end) * (1 + torch.cos(torch.tensor(current_epoch * 3.1415 / self.total_epochs)))
        T=15
        # 动态权重计算（线性衰减）
        soft_weight = max(0.7 - 0.4 * (current_epoch / self.total_epochs), 0.3)  # 70%->30%
        # # 软目标处理（带温度调节）
        soft_target = torch.softmax(teacher_logits / T, dim=-1)
        soft_loss = self.soft_loss(
            torch.log_softmax(student_logits / T, dim=-1),
            soft_target.detach()
        ) * (T ** 2)
        # soft_loss = self.hard_loss(teacher_logits, ps1)
        # soft_loss = torch.mean((teacher_logits - ps1)**2)
        # 硬目标处理
        hard_loss = self.hard_loss(student_logits, labels)
        # #置信度权重
        # teacher_probs = F.softmax(teacher_logits, dim=1)  # 计算概率分布
        # confidence = torch.gather(teacher_probs, dim=1, index=labels.unsqueeze(1)).squeeze(1)  # 提取真实标签位置上的概率
        # hard_loss = (hard_loss * confidence).mean()
        # soft_loss = soft_loss.sum(dim=1)
        # soft_loss = (soft_loss * confidence).mean()
        hard_weight = 1 - soft_weight
        # 组合损失
        total_loss = soft_weight * soft_loss + hard_weight * hard_loss
        return total_loss, soft_loss.item(), hard_loss.item()

def train(epoch, args, net, ju_dataloader, optimizer, loss_function, loss_function_kd, rank, warmup_scheduler=None, net_pretrained=None, last_layer=None):
    # ju_dataloader = loaddata(args.data, batch_size=8, set_name='train', shuffle=True, num_workers=args.num_workers)['train']
    start = time.time()

    net.train()
    for batch_index, (images, edge_images, labels) in enumerate(ju_dataloader['train']):
        images = images.to(rank, non_blocking=True)
        edge_images = edge_images.to(rank, non_blocking=True)
        labels = labels.to(rank, non_blocking=True)

        optimizer.zero_grad()
        outputs, output_ft = net(images, edge_images)
        pt = net_pretrained(images, edge_images)
        loss, soft_loss, hard_loss = loss_function(outputs, 1, pt, labels, epoch)
        loss.backward()
        for name, param in net.named_parameters():
            if param.grad is None:
                print(name)
        optimizer.step()

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * len(images) + len(images),
            total_samples=len(ju_dataloader['train'])
        ))

    avg_loss = loss / len(ju_dataloader['train'])

    if epoch <= args.warm:
        warmup_scheduler.step()

    wandb.log({"train_loss": avg_loss})

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
# def eval_training(epoch=0, tb=True):
def eval_training(epoch, net, test_loader, loss_function, rank):
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
    for (rgb_images, edge_images, labels) in test_loader['test']:
        rgb_images = rgb_images.to(rank, non_blocking=True)
        edge_images = edge_images.to(rank, non_blocking=True)
        labels = labels.to(rank, non_blocking=True)

        # 前向传播
        outputs, _ = net(rgb_images, edge_images)
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
            wandb.log({f"{label}_accuracy": accuracy, f"{label}_f1": f1, f"{label}_precision": precision, f"{label}_recall": recall})

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
    return accuracy


def main(rank, world_size, args):

    # 初始化分布式环境
    setup(rank, world_size)
    # 设置当前 GPU
    torch.cuda.set_device(rank)
    # 加载模型
    net = get_network(args)
    net = net.to(rank)
    # 使用 DDP 包装模型
    net = DDP(net, device_ids=[rank], find_unused_parameters=True)

    loss_function_eval = nn.CrossEntropyLoss()
    # loss_function = FocalLoss()
    loss_function = DynamicWeightedLoss()
    loss_function_kd = nn.KLDivLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-1)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(loaddata(args.data, batch_size=16, set_name='train', shuffle=True, num_workers=args.num_workers)['train'])
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    #dataset
    # 加载数据
    train_loader = loaddata_edge(args.data, batch_size=args.b, set_name='train', shuffle=False, num_workers=args.num_workers, rank=rank, world_size=world_size)
    test_loader = loaddata_edge(args.data, batch_size=args.b, set_name='test', shuffle=False, num_workers=args.num_workers, rank=rank, world_size=world_size)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    wandb.init(project="+aug-KD-DGCFconvnext-DGCFresnet18-diffus180-v1")

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

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

    # 加载预训练模型（例如ResNet）
    net_pretrained = DGCF_ConvNeXt()
    net_pretrained = net_pretrained.to(rank)
    pretrained_weights = torch.load("/home/zhuy/projects_pyc/pytorch-baseline/checkpoint/DGCF_ConvNeXt/Friday_07_March_2025_20h_04m_28s/DGCF_ConvNeXt-100-regular.pth")  #"/home/zhuy/projects_pyc/pytorch-baseline/checkpoint/DGCF_Resnet/resnet18_acc=90_new/DGCF_Resnet-100-regular.pth"
    new_weights = {k.replace("module.", ""): v for k, v in pretrained_weights.items()}
    net_pretrained.load_state_dict(new_weights, strict=False)
    # 冻结所有层，包括最后一层
    for param in net_pretrained.parameters():
        param.requires_grad = False
    net_pretrained.eval()

    # ### 2. 提取最后一层
    # last_layer = torch.nn.Sequential(
    #     net_pretrained.avg_pool,    # 全局平均池化
    #     torch.nn.Flatten(),         # 展平操作
    #     net_pretrained.fc           # 全连接层
    # )
    # for param in last_layer.parameters():
    #     param.requires_grad = False  # 冻结参数
    # last_layer.eval()  # 设置为评估模式

    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        # train(epoch, args)
        train(epoch, args, net, train_loader, optimizer, loss_function, loss_function_kd, rank, warmup_scheduler, net_pretrained, last_layer=None)
        # acc = eval_training(epoch)
        if rank == 0 and epoch % 5 ==0:  # 只在主进程上验证
            acc = eval_training(epoch, net, test_loader, loss_function_eval, rank)

        #start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

    # 清理分布式环境
    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='DGCF_Resnet18', help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-num_workers', type=int, default=8)
    parser.add_argument('-b', type=int, default=24, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=5, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=5e-4, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-data', type=str, default='/disk2/zhuy/PV/EL/masked_EL/baseline_dataset/')
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
