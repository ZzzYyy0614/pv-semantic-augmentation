""" helper function

author baiyu
"""
import os
import sys
import re
import datetime
import cv2
import numpy
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_network(args):
    """ return given network
    """

    if args.net == 'vgg16':
        from models.vgg import vgg16_bn
        net = vgg16_bn()
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn()
    elif args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn()
    elif args.net == 'vgg19':
        from models.vgg import vgg19_bn
        net = vgg19_bn()
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121()
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161()
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169()
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201()
    elif args.net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet()
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3()
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4()
    elif args.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2()
    elif args.net == 'xception':
        from models.xception import xception
        net = xception()
    elif args.net == 'resnet18':
        from models.resnet import resnet18
        net = resnet18()
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34()
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50()
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101()
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152()
    elif args.net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18()
    elif args.net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34()
    elif args.net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50()
    elif args.net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101()
    elif args.net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152()
    elif args.net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50()
    elif args.net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101()
    elif args.net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152()
    elif args.net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet()
    elif args.net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2()
    elif args.net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet()
    elif args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet()
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2()
    elif args.net == 'nasnet':
        from models.nasnet import nasnet
        net = nasnet()
    elif args.net == 'attention56':
        from models.attention import attention56
        net = attention56()
    elif args.net == 'attention92':
        from models.attention import attention92
        net = attention92()
    elif args.net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18()
    elif args.net == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34()
    elif args.net == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50()
    elif args.net == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101()
    elif args.net == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152()
    elif args.net == 'wideresnet':
        from models.wideresidual import wideresnet
        net = wideresnet()
    elif args.net == 'stochasticdepth18':
        from models.stochasticdepth import stochastic_depth_resnet18
        net = stochastic_depth_resnet18()
    elif args.net == 'stochasticdepth34':
        from models.stochasticdepth import stochastic_depth_resnet34
        net = stochastic_depth_resnet34()
    elif args.net == 'stochasticdepth50':
        from models.stochasticdepth import stochastic_depth_resnet50
        net = stochastic_depth_resnet50()
    elif args.net == 'stochasticdepth101':
        from models.stochasticdepth import stochastic_depth_resnet101
        net = stochastic_depth_resnet101()
    elif args.net == 'DGCF_Resnet':
        from models.resnet import DGCF_Resnet
        net = DGCF_Resnet()
    elif args.net == 'DGCF_Resnet18':
        from models.resnet import DGCF_Resnet18
        net = DGCF_Resnet18()
    elif args.net == 'ConvNeXt':
        from models.convnext import ConvNeXt
        net = ConvNeXt()
    elif args.net == 'DGCF_ConvNeXt':
        from models.convnext import DGCF_ConvNeXt
        net = DGCF_ConvNeXt()
    elif args.net == 'SwinT':
        from models.swinT_small import swinT
        net = swinT()
    elif args.net == 'DGCF_swinT':
        from models.swinT_small import DGCF_swinT
        net = DGCF_swinT()

    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    if args.gpu: #use_gpu
        net = net.cuda()

    return net


def get_training_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        #transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_training_loader

def get_test_dataloader(mean, std, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return cifar100_test_loader

from typing import Any, Callable, cast, Dict, List, Optional, Tuple
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")
# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except OSError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)
def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)
class add_imagefoder(datasets.DatasetFolder):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        # sample = self.loader(path)
        # 使用Sobel算子进行边缘检测
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_edges = cv2.magnitude(sobelx, sobely)
        sobel_edges = Image.fromarray(sobel_edges)
        if self.transform is not None:
            sample = self.transform(sobel_edges)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


# 自定义高斯噪声 transform 类
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        # 生成与输入张量相同形状的高斯噪声
        noise = torch.randn(tensor.size()) * self.std + self.mean
        # 将噪声加到图像上
        return torch.clamp(tensor + noise, 0., 1.)  # 保证图像值在 [0, 1] 范围内
    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std})'

def loaddata(root, batch_size, set_name, shuffle, num_workers, rank=None, world_size=None):
    #常见的数据增强方法，可增加：如翻转、平移
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.RandomHorizontalFlip(p=0.5),  # 以 50% 概率进行水平翻转
            # transforms.RandomAffine(degrees=10,  # 随机旋转在 [-10°, 10°] 范围内
            #                         translate=(0.1, 0.1)),  # 平移占宽高的 10%
            transforms.ToTensor(),
            # AddGaussianNoise(mean=0., std=0.01),  # 添加高斯噪声
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # transforms.Normalize([0.485], [0.229])
        ]),
         #常见的数据增强方法，可增加
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # transforms.Normalize([0.485], [0.229])
        ]),
        'rest': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # transforms.Normalize([0.485], [0.229])
        ]),
        'test_v2_rest': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # transforms.Normalize([0.485], [0.229])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(root, x), data_transforms[x]) for x in [set_name]}   #add_imagefoder
    # num_workers=0 if CPU else =1
    # dataset_loaders = {x: torch.utils.data.DataLoader(image_datasets[x],
    #                                                   batch_size=batch_size,
    #                                                   shuffle=shuffle, num_workers=num_workers) for x in [set_name]}
    if set_name == 'train' and rank is not None and world_size is not None:
        # 训练集需要使用 DistributedSampler
        train_sampler = DistributedSampler(image_datasets[set_name], num_replicas=world_size, rank=rank)
        dataset_loaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True) for x in [set_name]}
    else:
        # 测试集可以直接使用标准的 DataLoader
        dataset_loaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True) for x in [set_name]}
    return dataset_loaders

class RGBEdgeDataset(Dataset):
    def __init__(self, rgb_dir, edge_dir, rgb_transform=None, edge_transform=None):
        """
        初始化数据集
        :param rgb_dir: RGB 图像的文件夹路径
        :param edge_dir: 边缘图像的文件夹路径
        :param transform: 数据增强和预处理
        """
        self.rgb_dir = rgb_dir
        self.edge_dir = edge_dir
        self.rgb_transform = rgb_transform
        self.edge_transform = edge_transform
        
        # 获取所有类别的文件夹
        self.classes = sorted(os.listdir(rgb_dir))
        
        # 存储每个类别的 RGB 和边缘图像路径
        self.rgb_paths = []
        self.edge_paths = []
        self.labels = []
        
        for label, class_name in enumerate(self.classes):
            # RGB 图像路径
            rgb_class_dir = os.path.join(rgb_dir, class_name)
            rgb_filenames = sorted(os.listdir(rgb_class_dir))
            self.rgb_paths.extend([os.path.join(rgb_class_dir, fname) for fname in rgb_filenames])
            
            # 边缘图像路径
            edge_class_dir = os.path.join(edge_dir, class_name)
            edge_filenames = sorted(os.listdir(edge_class_dir))
            self.edge_paths.extend([os.path.join(edge_class_dir, fname) for fname in edge_filenames])
            
            # 标签
            self.labels.extend([label] * len(rgb_filenames))
        
        # 确保 RGB 图像和边缘图像的文件名一一对应
        assert len(self.rgb_paths) == len(self.edge_paths), "RGB 和边缘图像的数量不匹配！"

    def __len__(self):
        """返回数据集的大小"""
        return len(self.rgb_paths)

    def __getitem__(self, idx):
        """
        根据索引加载 RGB 图像和对应的边缘图像
        :param idx: 数据索引
        :return: RGB 图像、边缘图像和标签
        """
        # 加载 RGB 图像
        rgb_image = Image.open(self.rgb_paths[idx]).convert('RGB')  # 确保是 3 通道 RGB 图像
        
        # 加载边缘图像
        edge_image = Image.open(self.edge_paths[idx]).convert('L')  # 转换为灰度图像（单通道）
        # # 反色处理
        # edge_image = ImageOps.invert(edge_image)
        
        # 数据增强和预处理
        if self.rgb_transform:
            rgb_image = self.rgb_transform(rgb_image)
            edge_image = self.edge_transform(edge_image)
        
        # 标签
        label = self.labels[idx]
        
        return rgb_image, edge_image, label


def loaddata_edge(root, batch_size, set_name, shuffle, num_workers, rank=None, world_size=None):
    # 数据增强方法
    data_transforms = {
        'train': {
            'rgb': transforms.Compose([
                transforms.Resize((224, 224)),
                # transforms.RandomHorizontalFlip(p=0.5),  # 以 50% 概率进行水平翻转
                # transforms.RandomAffine(degrees=10,  # 随机旋转在 [-10°, 10°] 范围内
                #                     translate=(0.1, 0.1)),  # 平移占宽高的 10%
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'edge': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485], [0.229])  # 单通道归一化
            ]),
            'rgb-edge': A.Compose([
                A.Resize(height=224, width=224),  # 调整图像和边缘图的大小
                A.Rotate(limit=10, p=1),  # 随机旋转，角度范围在-30到30度之间
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=0, p=1),  # 随机平移
                A.Normalize(  # 对RGB图像进行归一化
                    mean=[0.485, 0.456, 0.406],  # ImageNet的均值
                    std=[0.229, 0.224, 0.225],   # ImageNet的标准差
                    max_pixel_value=255.0,       # 像素值范围
                ),
                # 对边缘图标准化（单通道）
                A.Normalize(
                    mean=[0.5],  # 单通道均值（例如 0.5）
                    std=[0.5],   # 单通道标准差（例如 0.5）
                    max_pixel_value=255.0,
                    always_apply=True,
                    p=1.0,
                ),
                ToTensorV2(),  # 将图像和边缘图转换为张量
            ], additional_targets={'edge_map': 'image'})  # 指定边缘图作为额外目标
        }
    }

    # 根据 set_name 设置对应的文件夹
    if set_name == 'train':
        rgb_dir = os.path.join(root, 'train')  # 训练集 RGB 图像路径
        # rgb_transform = data_transforms['train']['rgb']
        rgb_transform = data_transforms['train']['rgb']
        edge_dir = os.path.join(root, 'train_sobeledge')  # 训练集边缘图像路径
        edge_transform = data_transforms['train']['edge']
    elif set_name == 'test':
        rgb_dir = os.path.join(root, 'test')  # 验证集 RGB 图像路径
        # rgb_transform = data_transforms['train']['rgb']
        rgb_transform = data_transforms['train']['rgb']
        edge_dir = os.path.join(root, 'test_sobeledge')  # 验证集边缘图像路径
        edge_transform = data_transforms['train']['edge']
    else:
        raise ValueError("set_name 必须是 'train' 或 'val'")

    # 创建自定义数据集
    image_datasets = {set_name: RGBEdgeDataset(rgb_dir, edge_dir, rgb_transform=rgb_transform, edge_transform=edge_transform)}

    # 创建 DataLoader
    if set_name == 'train' and rank is not None and world_size is not None:
        # 训练集需要使用 DistributedSampler
        train_sampler = DistributedSampler(image_datasets[set_name], num_replicas=world_size, rank=rank)
        dataset_loaders = {set_name: DataLoader(
            image_datasets[set_name],
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True
        )}
    else:
        # 验证集可以直接使用标准的 DataLoader
        dataset_loaders = {set_name: DataLoader(
            image_datasets[set_name],
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )}

    return dataset_loaders

def resnet18_infer_data(root_dir):
    instances = []
    for root, dirs, fnames in sorted(os.walk(os.path.join(root_dir,'train'))):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            for fname in sorted(os.listdir(dir_path)):
                path = os.path.join(dir_path, fname)
                instances.append(path)
    return instances
class resnet18_infer():
    def __init__(self, root, resnet18_weight):
        self.root = root
        self.weight = resnet18_weight
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            transforms.Normalize([0.485], [0.229])
        ])
        from models.resnet import resnet18
        self.net = resnet18()
        self.net.load_state_dict(torch.load(self.weight))
        self.images = resnet18_infer_data(root)
    def __getitem__(self, index):
        self.net.eval()
        image_path = self.images[index]
        image = Image.open(image_path).convert('RGB')
        # # 使用Sobel算子进行边缘检测
        # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        # sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        # sobel_edges = cv2.magnitude(sobelx, sobely)
        # image = Image.fromarray(sobel_edges)
        image = self.transform(image)
        sample = torch.tensor(image)
        output = self.net(sample[None,:])
        output = F.softmax(output, dim=1)
        target = output.detach().squeeze(0)
        return sample, target
    def __len__(self):
        return len(self.images)

def resnet18_txt_data(txt_dir):
    instances = []
    with open(txt_dir, 'r') as f:
        lines = f.readlines()
        for line in lines:
            img_path = line.split(';')[0]
            target = line.split(';')[1]
            instances.append([img_path, target])
    return instances
class resnet18_txt_infer():
    def __init__(self, root):
        self.root = root
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # transforms.Normalize([0.485], [0.229])
        ])
        self.files = resnet18_txt_data("/disk1/zhuy/PV/EL/masked_EL/baseline_dataset/train.txt")
    def __getitem__(self, index):
        image_path = self.files[index][0]
        image = Image.open(image_path).convert('RGB')
        # # 使用Sobel算子进行边缘检测
        # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        # sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        # sobel_edges = cv2.magnitude(sobelx, sobely)
        # image = Image.fromarray(sobel_edges)
        image = self.transform(image)
        sample = torch.tensor(image)
        target = torch.tensor(eval(self.files[index][1]))
        return sample, target
    def __len__(self):
        return len(self.files)

def loaddata2(root, batch_size, set_name, shuffle, num_workers):
    dataset_loaders = torch.utils.data.DataLoader(resnet18_txt_infer(root), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataset_loaders

def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]