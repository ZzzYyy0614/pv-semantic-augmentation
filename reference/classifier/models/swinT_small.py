import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def swinT():
    return SwinTransformer()

def DGCF_swinT():
    return DGCF_swinT()

class PatchEmbed(nn.Module):
    """图像分块嵌入层"""
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.proj = nn.Conv2d(in_chans, embed_dim, 
                            kernel_size=patch_size, 
                            stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, C, H, W]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x

##-----不适用边缘对齐的PatchMerging-----##
# class PatchMerging(nn.Module):
#     """特征图下采样层"""
#     def __init__(self, dim):
#         super().__init__()
#         self.norm = nn.LayerNorm(4 * dim)
#         self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

#     def forward(self, x):
#         B, L, C = x.shape
#         H = W = int(L ** 0.5)
        
#         # 重排为空间特征
#         x = x.view(B, H, W, C)
        
#         # 2x2邻域采样
#         x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
#         x1 = x[:, 1::2, 0::2, :]
#         x2 = x[:, 0::2, 1::2, :]
#         x3 = x[:, 1::2, 1::2, :]
        
#         # 通道拼接
#         x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
#         x = x.view(B, -1, 4 * C)  # [B, (H/2)*(W/2), 4*C]
        
#         # 线性变换
#         x = self.norm(x)
#         x = self.reduction(x)
#         return x

##-----适用边缘对齐的PatchMerging-----##
class PatchMerging(nn.Module):
    """特征图下采样层（支持奇数尺寸）"""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        B, H, W, C = x.shape

        # 确保高度和宽度是偶数
        if H % 2 != 0 or W % 2 != 0:
            # 如果尺寸为奇数，填充到偶数
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
            H, W = x.shape[1], x.shape[2]

        # 划分特征图
        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        # 拼接特征
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, (H/2)*(W/2), 4*C]

        # 归一化和线性变换
        x = self.norm(x)
        x = self.reduction(x)
        return x

class WindowAttention(nn.Module):
    """基于窗口的自注意力机制"""
    def __init__(self, dim, num_heads=8, window_size=7, shift_size=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 定义 qkv 映射
        self.qkv = nn.Linear(dim, dim * 3)  # 将输入特征映射到 Q, K, V 空间

        # 相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads))
        
        # 生成相对位置索引
        coords = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords, coords))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        # 初始化参数
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x):
        B, H, W, C = x.shape  # 输入特征图的形状 [B, H, W, C]
        N = self.window_size * self.window_size  # 每个窗口的 token 数
        num_windows = (H // self.window_size) * (W // self.window_size)  # 窗口总数
        B_ = B * num_windows  # 窗口划分后的批次大小

        # 移位窗口
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # 划分窗口
        x_windows = window_partition(shifted_x, self.window_size)  # [B_, window_size, window_size, C]
        x_windows = x_windows.view(-1, N, C)  # [B_, N, C]

        # 自注意力计算
        qkv = self.qkv(x_windows)  # [B_, N, 3 * C]
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads)  # [B_, N, 3, num_heads, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B_, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # 分离 Q, K, V

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B_, num_heads, N, N]

        # 添加相对位置偏置
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size**2, self.window_size**2, -1)
        attn = attn + relative_position_bias.permute(2, 0, 1).unsqueeze(0)  # [B_, num_heads, N, N]

        # 计算注意力输出
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)  # [B_, N, C]

        # 合并窗口
        x = x.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(x, self.window_size, H, W)  # [B, H, W, C]

        # 逆移位
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        return x

class SwinBlock(nn.Module):
    """基础Swin Transformer块"""
    def __init__(self, dim, num_heads, window_size=7, shift_size=0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, num_heads=num_heads, window_size=window_size,
            shift_size=shift_size
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = x.view(B, H, W, C)
        
        # 窗口注意力
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + x
        
        # MLP
        x = x.view(B, L, C)
        shortcut = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + x
        return x

class SwinStage(nn.Module):
    """完整的Swin阶段（包含多个块和下采样）"""
    def __init__(self, dim, depth, num_heads, window_size, downsample=True):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(dim=dim, num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2)
            for i in range(depth)
        ])
        if downsample:
            self.downsample = PatchMerging(dim)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class SwinTransformer(nn.Module):
    """完整的Swin Transformer模型"""
    def __init__(self, num_classes=5, 
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7):
        super().__init__()
        
        # 参数设置
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # 输入处理
        self.patch_embed = PatchEmbed(embed_dim=embed_dim)
        self.pos_drop = nn.Dropout(p=0.1)
        
        # 构建各阶段
        self.stage1 = SwinStage(embed_dim, depths[0], num_heads[0], window_size)
        self.stage2 = SwinStage(embed_dim*2, depths[1], num_heads[1], window_size)
        self.stage3 = SwinStage(embed_dim*4, depths[2], num_heads[2], window_size)
        self.stage4 = SwinStage(embed_dim*8, depths[3], num_heads[3], window_size, downsample=False)
        
        # 分类头
        self.norm = nn.LayerNorm(embed_dim * 8)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(embed_dim * 8, num_classes)

    def forward(self, x):
        # 输入嵌入
        x = self.patch_embed(x)  # [B, L, C]
        x = self.pos_drop(x)
        
        # 逐阶段处理
        x = self.stage1(x)  # [B, L, C]
        x = self.stage2(x)   # [B, L/4, 2C]
        x = self.stage3(x)   # [B, L/16, 4C]
        x = self.stage4(x)  # [B, L/64, 8C]
        
        # 分类处理
        x = self.norm(x)                # [B, L/64, 8C]
        x = self.avgpool(x.transpose(1, 2))  # [B, 8C, 1]
        x = torch.flatten(x, 1)         # [B, 8C]
        x = self.head(x)                # [B, num_classes]
        return x

def window_partition(x, window_size):
    """
    将特征图划分为窗口
    参数:
        x: (B, H, W, C)
        window_size (int): 窗口大小
    返回:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    将窗口还原为特征图
    参数:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): 窗口大小
        H (int): 特征图高度
        W (int): 特征图宽度
    返回:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class EdgeBranch(nn.Module):
    """边缘分支（第 0 层输出为 192x28x28，与主干 Stage 1 对齐）"""
    def __init__(self, embed_dim=96):
        super().__init__()
        self.stages = nn.ModuleList([
            # Stage 0: [1, 224, 224] -> [192, 28, 28] (stride=8)
            nn.Sequential(
                nn.Conv2d(1, embed_dim*2, kernel_size=7, stride=8, padding=3),
                nn.BatchNorm2d(embed_dim*2),
                nn.ReLU(inplace=True)
            ),
            # Stage 1: [192, 28, 28] -> [384, 14, 14] (stride=2)
            nn.Sequential(
                nn.Conv2d(embed_dim*2, embed_dim*4, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(embed_dim*4),
                nn.ReLU(inplace=True)
            ),
            # Stage 2: [384, 14, 14] -> [768, 7, 7] (stride=2)
            nn.Sequential(
                nn.Conv2d(embed_dim*4, embed_dim*8, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(embed_dim*8),
                nn.ReLU(inplace=True)
            ),
            # Stage 3: [768, 7, 7] -> [768, 7, 7] (无下采样)
            nn.Sequential(
                nn.Conv2d(embed_dim*8, embed_dim*8, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(embed_dim*8),
                nn.ReLU(inplace=True)
            )
        ])

    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features

class DeformCrossModality(nn.Module):
    """可变形跨模态对齐模块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.align_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//16, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels//16, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, rgb_feat, edge_feat):
        aligned_rgb = self.align_conv(rgb_feat)
        edge_att = self.channel_att(edge_feat)
        return aligned_rgb * edge_att + edge_feat

class SimpleCrossModality(nn.Module):
    """极简跨模态融合模块"""
    def __init__(self, in_channels):
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels, 1),  # 1x1卷积压缩通道
            nn.ReLU(inplace=True))
    
    def forward(self, rgb_feat, edge_feat):
        fused = torch.cat([rgb_feat, edge_feat], dim=1)
        return self.fusion_conv(fused)

class DGCF_swinT(nn.Module):
    def __init__(self, num_classes=5, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7):
        super().__init__()
        self.embed_dim = embed_dim

        # 1. 主干网络（严格保持 Swin-T 原始结构）
        self.patch_embed = PatchEmbed(embed_dim=embed_dim)
        self.pos_drop = nn.Dropout(p=0.1)

        # Swin 阶段定义（通道数与原始 Swin-T 完全一致）
        self.stages = nn.ModuleList([
            SwinStage(
                dim=embed_dim * (2 ** i),  # 关键点：保持原始通道数
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size
            ) for i in range(4)
        ])

        # 2. 边缘分支（对齐主干的通道和空间尺寸）
        self.edge_branch = EdgeBranch(embed_dim=embed_dim)

        # 3. 跨模态模块（输入通道与主干各阶段严格一致）
        self.dcm_blocks = nn.ModuleList([
            DeformCrossModality(
                in_channels=embed_dim * (2 ** (i+1)),  # 使用当前阶段的通道数
                out_channels=embed_dim * (2 ** (i+1))
            ) for i in range(4)
        ])
        self.dgf_blocks = nn.ModuleList([
            SimpleCrossModality(embed_dim * (2 ** (i+1))) for i in range(4)
        ])

        # 分类头（保持原始输出维度）
        self.norm = nn.LayerNorm(embed_dim * 8)  # 原始 Swin-T 最终维度是 embed_dim*8
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(embed_dim * 8, num_classes)

    def forward(self, x, edge):
        x = self.patch_embed(x)  # [B, L, C]
        x = self.pos_drop(x)
        edge_feats = self.edge_branch(edge)

        for i in range(4):
            x = self.stages[i](x)  # 主干处理（通道数保持原始 Swin-T 结构）

            if i < len(edge_feats):
                # 转换序列特征为空间特征
                B, L, C = x.shape
                H = W = int(L ** 0.5)
                x_spatial = x.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]

                # 严格验证通道和空间尺寸
                assert x_spatial.shape[1] == edge_feats[i].shape[1], \
                    f"Stage {i}通道不匹配: 主干 {x_spatial.shape[1]} vs 边缘 {edge_feats[i].shape[1]}"
                assert x_spatial.shape[2:] == edge_feats[i].shape[2:], \
                    f"Stage {i}空间尺寸不匹配: 主干 {x_spatial.shape[2:]} vs 边缘 {edge_feats[i].shape[2:]}"

                # 跨模态融合
                aligned_feat = self.dcm_blocks[i](x_spatial, edge_feats[i])
                fused_feat = self.dgf_blocks[i](x_spatial, aligned_feat)

                # 转换回序列并残差连接
                x = fused_feat.permute(0, 2, 3, 1).reshape(B, L, C) + x

        # 分类输出（保持原始 Swin-T 结构）
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return self.head(x)