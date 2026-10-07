import torch
import torch.nn as nn
import torch.nn.functional as F

def ConvNeXt():
    return ConvNeXt()

def DGCF_Resnet():
    return DGCF_ConvNeXt()

class ConvNeXt(nn.Module):
    def __init__(self, 
                 depths=[3, 3, 9, 3], 
                 dims=[96, 192, 384, 768],
                 num_classes=5):
        super().__init__()
        
        # ---------------------------- 主干网络 ----------------------------
        # 初始stem层 (4x4卷积实现patchify)
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        
        # 构建4个阶段
        self.stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        # 遍历每个阶段
        for i in range(4):
            # ---------------------------- 阶段块 ----------------------------
            # 每个阶段包含多个ConvNeXt块
            stage = nn.Sequential(
                *[ConvNeXtBlock(dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)
            
            # ---------------------------- 下采样层 ----------------------------
            # 前3个阶段后需要下采样（最后一个阶段后不需要）
            if i < 3:
                downsample = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
                )
                self.downsample_layers.append(downsample)

        # ---------------------------- 分类头 ----------------------------
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # 最终层归一化
        self.head = nn.Linear(dims[-1], num_classes)  # 分类头

        # ---------------------------- 初始化 ----------------------------
        self._init_weights()

    def _init_weights(self):
        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        # Stem层
        x = self.stem(x)
        
        # 各阶段处理
        for i in range(4):
            x = self.stages[i](x)     # ConvNeXt块处理
            if i < 3:                # 前3个阶段后下采样
                x = self.downsample_layers[i](x)
        return x

    def forward(self, x):
        # 特征提取
        x = self.forward_features(x)
        
        # 分类头
        x = self.norm(x.mean([-2, -1]))  # 全局平均池化+层归一化
        x = self.head(x)
        return x



class DGCF_ConvNeXt(nn.Module):
    def __init__(self, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], num_classes=5, use_edge=True):
        super().__init__()
        self.use_edge = use_edge
        
        # ConvNeXt主干配置
        self.depths = depths
        self.dims = dims
        
        # 原始ConvNeXt主干
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        
        # 边缘特征分支
        if use_edge:
            # 边缘分支的通道数与主分支对齐
            edge_dims = dims  # [48, 96, 192, 384],[dims[0]//2] + dims[:-1]
            
            self.edge_branch = nn.Sequential(
                # Stage 0: 对齐stem层（输出[B, 48, 56, 56]）
                nn.Sequential(
                    nn.Conv2d(1, edge_dims[0], 3, stride=4, padding=1),
                    LayerNorm(edge_dims[0], eps=1e-6, data_format="channels_first"),
                    nn.GELU()
                ),
                
                # 各阶段与主分支对齐
                *[self._make_edge_stage(edge_dims[i], edge_dims[i+1]) 
                 for i in range(len(edge_dims)-1)]
            )
            
            # 多尺度融合模块（通道数与主分支对齐）
            self.dcm_blocks = nn.ModuleList([
                DeformCrossModality(dims[0], edge_dims[0]),   # Stage 1
                DeformCrossModality(dims[1], edge_dims[1]),   # Stage 2
                DeformCrossModality(dims[2], edge_dims[2]),   # Stage 3
                DeformCrossModality(dims[3], edge_dims[3])    # Stage 4
            ])
            
            # 动态门控融合
            self.dgf_blocks = nn.ModuleList([
                SimpleCrossModality(dims[0]),
                SimpleCrossModality(dims[1]),
                SimpleCrossModality(dims[2]),
                SimpleCrossModality(dims[3])
            ])
        
        # ConvNeXt主干层
        self.stages = nn.ModuleList()
        dpth = depths
        for i in range(4):
            stage = nn.Sequential(
                *[ConvNeXtBlock(dims[i]) for _ in range(dpth[i])]
            )
            self.stages.append(stage)
            
        # 下采样层
        self.downsample_layers = nn.ModuleList()
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        
        # 分类头
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

    def _make_edge_stage(self, in_dim, out_dim):
        """边缘分支的单个阶段"""
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1),
            LayerNorm(out_dim, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
            LayerNorm(out_dim, eps=1e-6, data_format="channels_first"),
            nn.GELU()
        )

    def forward(self, x, y):
        # 主分支处理
        x = self.stem(x)  # [B, 96, 56, 56]
        
        # 边缘分支处理
        if self.use_edge:
            y = self.edge_branch[0](y)  # [B, 48, 56, 56]
            
            # 第一阶段融合
            x = self.stages[0](x)  # [B, 96, 56, 56]
            x = self.dgf_blocks[0](x, self.dcm_blocks[0](x, y))
            
            # 下采样并进入下一阶段
            for i in range(3):
                x = self.downsample_layers[i](x)  # 下采样
                x = self.stages[i+1](x)  # ConvNeXt块处理
                
                # 边缘分支下采样
                y = self.edge_branch[i+1](y)  # 更新边缘特征
                
                # 特征融合
                x = self.dgf_blocks[i+1](x, self.dcm_blocks[i+1](x, y))
        
        # 分类头
        x = self.norm(x.mean([-2, -1]))  # 全局平均池化+LayerNorm
        x = self.head(x)
        return x

# ConvNeXt基础块（简化版）
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        return residual + x
    
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
    
class LayerNorm(nn.Module):
    """支持 channels_first 模式的 LayerNorm"""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"Unsupported data format: {self.data_format}")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            # 标准 LayerNorm 实现
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # 针对 channels_first 模式的自定义实现
            u = x.mean(1, keepdim=True)  # 在通道维度计算均值
            s = (x - u).pow(2).mean(1, keepdim=True)  # 在通道维度计算方差
            x = (x - u) / torch.sqrt(s + self.eps)  # 归一化
            x = self.weight[:, None, None] * x + self.bias[:, None, None]  # 缩放和平移
            return x