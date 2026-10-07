"""resnet in pytorch



[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=5):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

def resnet18():
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])

def resnet34():
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3])

def resnet50():
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3])

def resnet101():
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3])

def resnet152():
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3])

def DGCF_Resnet():
    return DGCF_Res(BasicBlock, [2, 2, 2, 2])

def DGCF_Resnet18():
    return DGCF_Res18(BasicBlock, [2, 2, 2, 2])


class DGCF_Res(nn.Module):
    def __init__(self, block, num_block, num_classes=5, use_edge=True):
        super().__init__()
        self.use_edge = use_edge
        
        # 原始ResNet主干
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        
        # 边缘特征分支
        if use_edge:
            self.edge_branch = nn.Sequential(
                # Stage 1: 与 conv1 对齐（输出 [B, 64, 224, 224]）
                nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                # Stage 2: 与 conv2_x 对齐（输出 [B, 256, 224, 224]）
                self._make_edge_layer(64, 64, stride=1),  # 调整通道数，保持尺寸,res50是self._make_edge_layer(64, 256, stride=1)
                # Stage 3: 与 conv3_x 对齐（输出 [B, 512, 112, 112]）
                self._make_edge_layer(64, 128, stride=2),  # 下采样到 112x112，self._make_edge_layer(256, 512, stride=2)
                # Stage 4: 与 conv4_x 对齐（输出 [B, 1024, 56, 56]）
                self._make_edge_layer(128, 256, stride=2),  # 下采样到 56x56
                # Stage 5: 与 conv5_x 对齐（输出 [B, 2048, 28, 28]）
                self._make_edge_layer(256, 512, stride=2),   # 下采样到 28x28
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Linear(256, num_classes)
            )
            
            # 多尺度融合模块（通道数与 ResNet50 各阶段对齐）
            self.dcm_blocks = nn.ModuleList([
                DeformCrossModality(64, 64),   # 对应 conv2_x 的 256 通道
                DeformCrossModality(128, 128),   # 对应 conv3_x 的 512 通道
                DeformCrossModality(256, 256), # 对应 conv4_x 的 1024 通道
                DeformCrossModality(512, 512)  # 对应 conv5_x 的 2048 通道
            ])
            
            # 动态门控融合（通道数与 ResNet50 各阶段对齐）
            self.dgf_blocks = nn.ModuleList([
                SimpleCrossModality(64),
                SimpleCrossModality(128),
                SimpleCrossModality(256),
                SimpleCrossModality(512)
            ])
        
        # ResNet主干层（融合后）
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        
        # 分类头
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)   #512 * block.expansion

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            # 添加残差块
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        
        return nn.Sequential(*layers)
    
    def _make_edge_layer(self, in_c, out_c, stride):
        """边缘分支的下采样层（与 ResNet50 各阶段对齐）"""
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, y):
        # 基础特征提取
        x = self.conv1(x)         # [B,64,H,W]
        # 边缘特征提取
        y = self.edge_branch[0:4](y)  # [B,64,H,W]
        # 多尺度特征融合
        x = self.conv2_x(x)      # [B,64,H,W]
        x = self.dgf_blocks[0](x, self.dcm_blocks[0](x, y))  # [B,128,H/2,W/2]
        x = self.conv3_x(x)  # [B,128,H/2,W/2]
        y = self.edge_branch[4](y)        # [B,128,H/2,W/2]
        x = self.dgf_blocks[1](x, self.dcm_blocks[1](x, y))  # [B,256,H/4,W/4]
        x = self.conv4_x(x)  # [B,256,H/4,W/4]
        y = self.edge_branch[5](y)        # [B,256,H/4,W/4]
        x = self.dgf_blocks[2](x, self.dcm_blocks[2](x, y))  # [B,512,H/8,W/8]
        x = self.conv5_x(x)  # [B,512,H/8,W/8]
        # 分类
        x = self.avg_pool(x)
        x = self.fc(x.view(x.size(0), -1))
        return x
    
class DGCF_Res18(nn.Module):
    def __init__(self, block, num_block, num_classes=5, use_edge=True):
        super().__init__()
        self.use_edge = use_edge
        
        # 原始ResNet主干
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        
        # 边缘特征分支
        if use_edge:
            self.edge_branch = nn.Sequential(
                # Stage 1: 与 conv1 对齐（输出 [B, 64, 224, 224]）
                nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                # Stage 2: 与 conv2_x 对齐（输出 [B, 256, 224, 224]）
                self._make_edge_layer(64, 64, stride=1),  # 调整通道数，保持尺寸,res50是self._make_edge_layer(64, 256, stride=1)
                # Stage 3: 与 conv3_x 对齐（输出 [B, 512, 112, 112]）
                self._make_edge_layer(64, 128, stride=2),  # 下采样到 112x112，self._make_edge_layer(256, 512, stride=2)
                # Stage 4: 与 conv4_x 对齐（输出 [B, 1024, 56, 56]）
                self._make_edge_layer(128, 256, stride=2),  # 下采样到 56x56
                # # Stage 5: 与 conv5_x 对齐（输出 [B, 2048, 28, 28]）
                # self._make_edge_layer(256, 512, stride=2),   # 下采样到 28x28
                # nn.AdaptiveAvgPool2d((1, 1)),
                # nn.Linear(256, num_classes)
            )
            
            # 多尺度融合模块（通道数与 ResNet50 各阶段对齐）
            self.dcm_blocks = nn.ModuleList([
                DeformCrossModality(64, 64),   # 对应 conv2_x 的 256 通道
                DeformCrossModality(128, 128),   # 对应 conv3_x 的 512 通道
                DeformCrossModality(256, 256), # 对应 conv4_x 的 1024 通道
            ])
            
            # 动态门控融合（通道数与 ResNet50 各阶段对齐）
            self.dgf_blocks = nn.ModuleList([
                SimpleCrossModality(64),
                SimpleCrossModality(128),
                SimpleCrossModality(256)
            ])

            # 统一门控融合模块（每个阶段对应一个）
            self.gate_fusions = nn.ModuleList([
                SimpleGateFusion(64, 64, 64),    # 对应conv2_x
                SimpleGateFusion(128, 128, 128), # 对应conv3_x
                SimpleGateFusion(256, 256, 256)  # 对应conv4_x
            ])
        
        # ResNet主干层（融合后）
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        
        # 分类头
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)   #512 * block.expansion

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            # 添加残差块
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        
        return nn.Sequential(*layers)
    
    def _make_edge_layer(self, in_c, out_c, stride):
        """边缘分支的下采样层（与 ResNet50 各阶段对齐）"""
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, y):
        # 基础特征提取
        x = self.conv1(x)         # [B,64,H,W]
        
        # 边缘特征提取
        y = self.edge_branch[0:3](y)  # [B,64,H,W]
        
        # 多尺度特征融合
        x = self.conv2_x(x)      # [B,64,H,W]
        
        x = self.gate_fusions[0](x, y)

        x = self.conv3_x(x)  # [B,128,H/2,W/2]

        y = self.edge_branch[4](y)        # [B,128,H/2,W/2]
        
        x = self.gate_fusions[1](x, y)
        x = self.conv4_x(x)  # [B,256,H/4,W/4]

        y = self.edge_branch[5](y)        # [B,256,H/4,W/4]
        
        x = self.gate_fusions[2](x, y)   # 融合
        x = self.conv5_x(x)  # [B,512,H/8,W/8]

        # 分类
        x_cls = self.avg_pool(x)
        x_cls = self.fc(x_cls.view(x_cls.size(0), -1))
        return x_cls
    

class DeformCrossModality(nn.Module):
    """模态对齐模块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 普通卷积
        self.align_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 保持原有的通道注意力机制
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//16, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels//16, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, rgb_feat, edge_feat):
        # 简化后的对齐过程
        aligned_rgb = self.align_conv(rgb_feat)  # 普通卷积对齐
        edge_att = self.channel_att(edge_feat)   # 通道注意力
        return aligned_rgb * edge_att + edge_feat  # 特征融合

class DynamicGateFusion(nn.Module):
    """动态门控融合模块"""
    def __init__(self, channels):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Conv2d(channels*2, channels//4, 1),
            nn.ReLU(),
            nn.Conv2d(channels//4, 2, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, rgb_feat, aligned_feat):
        gate = self.gate_net(torch.cat([rgb_feat, aligned_feat], dim=1))
        return gate[:,0:1]*rgb_feat + gate[:,1:2]*aligned_feat

class SimpleCrossModality(nn.Module):
    """极简跨模态融合模块"""
    def __init__(self, in_channels):
        super().__init__()
        # 直接融合通道
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, 1),  # 1x1卷积压缩通道
            nn.ReLU(inplace=True))
    
    def forward(self, rgb_feat, edge_feat):
        fused = torch.cat([rgb_feat, edge_feat], dim=1)
        return self.fusion_conv(fused)

class ConcatFusion(nn.Module):
    """直接拼接融合模块"""
    def __init__(self, channels):
        super().__init__()
        # 简化融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(channels*2, channels, 1),  # 保持通道数不变
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True))
    
    def forward(self, rgb_feat, aligned_feat):
        combined = torch.cat([rgb_feat, aligned_feat], dim=1)
        return self.fusion(combined)
    
class SimpleGateFusion(nn.Module):
    """极简门控融合模块（包含维度对齐）"""
    def __init__(self, rgb_channels, edge_channels, out_channels):
        """
        :param rgb_channels: RGB特征通道数
        :param edge_channels: Edge特征通道数
        :param out_channels: 输出通道数
        """
        super().__init__()
        
        # RGB特征对齐
        self.rgb_align = nn.Sequential(
            nn.Conv2d(rgb_channels, out_channels, 1),  # 1x1卷积对齐通道
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Edge特征对齐
        self.edge_align = nn.Sequential(
            nn.Conv2d(edge_channels, out_channels, 1),  # 1x1卷积对齐通道
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # self.local_att = nn.Sequential(
        #     nn.Conv2d(edge_channels, edge_channels, kernel_size=1, stride=1, padding=0),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(edge_channels, edge_channels, kernel_size=1, stride=1, padding=0)
        # )
        # self.global_att = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(edge_channels, edge_channels, kernel_size=1, stride=1, padding=0),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(edge_channels, edge_channels, kernel_size=1, stride=1, padding=0)
        # )
        self.sigmoid = nn.Sigmoid()
        # 轻量门控网络
        self.gate_net = nn.Sequential(
            nn.Conv2d(2*out_channels, out_channels//4, 3, padding=1),  # 保持空间维度
            nn.ReLU(),
            nn.Conv2d(out_channels//4, 2, 3, padding=1),               # 输出双通道权重
            nn.Softmax(dim=1)
        )

    def forward(self, rgb_feat, edge_feat):
        # Step 2: 生成门控权重
        gate_input = torch.cat([rgb_feat, edge_feat], dim=1)  # [B, 2*out_c, H, W]
        gate = self.gate_net(gate_input)          # [B, 2, H, W]
        # Step 3: 加权融合
        return gate[:,0:1]*rgb_feat + gate[:,1:2]*edge_feat

    # def forward(self, rgb_feat, edge_feat):
    #     xa = rgb_feat + edge_feat
    #     xl = self.local_att(xa)
    #     xg = self.global_att(xa)
    #     xlg = xl + xg
    #     wei = self.sigmoid(xlg)
    #     xo = 2 * rgb_feat * wei + 2 * edge_feat * (1 - wei)
    #     return xo