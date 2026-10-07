# PV-Semantic-Edge 中文说明

本项目对应论文 [Semantic-aware data augmentation with edge priors for PV module defect recognition using EL images](https://doi.org/10.1016/j.solener.2025.114269)，包含草图条件图像生成、语义监督和边缘融合缺陷识别的实现。

![方法与代码结构](../assets/framework.svg)

## 方法与模型

训练分为三个部分：

1. **生成模型**：VQ 编码器分别编码 EL 图像和缺陷区域草图，草图特征与带噪图像特征拼接，类别信息通过交叉注意力输入 Sketch-LDM。
2. **语义估计器**：在真实训练图像上学习类别概率，识别器训练时保持参数冻结。
3. **缺陷识别器**：在三个尺度融合图像特征与 Sobel 边缘特征，通过交叉熵和温度缩放 KL 损失学习真实、生成图像。

识别模型支持 ResNet18、ConvNeXt-Tiny 和 Swin-T。默认使用门控融合，另外提供 additive 和 concat 融合供扩展消融使用。

## 安装

项目使用 Python 3.10–3.12、PyTorch 2.5.x 和 torchvision 0.20.x。Windows CPU 环境可按以下步骤安装：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
```

GPU 环境先安装对应 CUDA 版本的 PyTorch。所有 `pvaug` 命令也可以通过 `python -m pvaug` 执行。

运行小规模示例：

```bash
pvaug smoke --output runs/example
```

示例依次训练语义估计器、小型生成模型和识别器，并输出评估及 Grad-CAM。它使用模拟图像和随机小型 VQ 编码器，用于检查程序运行，指标不能作为论文复现结果。

## 训练配置

数据和训练权重尚未包含在仓库中，数据发布信息后续补充。CSV 接口为：

```text
image,label,split,domain,mask,edge,group
```

图像路径相对于 CSV 所在目录。生成模型需要人工区域标注。正常类别使用 `defect_free`；旧权重的类别顺序需要与其原始训练顺序一致。

| 配置 | 用途 |
|:--|:--|
| `configs/tasks/estimator.yaml` | 真实训练集上的语义估计器 |
| `configs/tasks/sketch_ldm.yaml` | 草图条件生成模型 |
| `configs/tasks/recognizer.yaml` | 生成数据与语义监督下的识别器 |
| `configs/pipelines/paper.yaml` | 完整训练、合成和评估流程 |

接入清单、类别顺序、草图及 VQ 权重后，先查看实际使用的参数：

```bash
pvaug config --config configs/tasks/estimator.yaml
pvaug pipeline --config configs/pipelines/paper.yaml --dry-run
```

执行完整流程，或单独训练一个模型：

```bash
pvaug pipeline --config configs/pipelines/paper.yaml
pvaug run --config configs/tasks/estimator.yaml
```

配置从 `configs/base/` 继承公共设置，可直接覆盖单个参数：

```bash
pvaug run --config configs/tasks/recognizer.yaml --set model.backbone=convnext_tiny --set experiment.output=runs/convnext
```

每次训练保存配置、日志和检查点。恢复已有训练：

```bash
pvaug run --config configs/tasks/estimator.yaml --resume runs/estimator/last.pt
pvaug pipeline --config configs/pipelines/paper.yaml --resume
```

训练从 epoch 边界恢复。完整流程恢复时会检查阶段输入和产物；采样阶段暂不支持部分恢复。详细设置见 [experiments.md](experiments.md)。

## 评估与消融

```bash
pvaug eval --checkpoint runs/paper/recognizer/best.pt --manifest data/real.csv --output runs/test
pvaug predict --checkpoint runs/paper/recognizer/best.pt --input data/example.png --output runs/predict --gradcam
pvaug sweep --config configs/sweeps/fusion.yaml --dry-run
pvaug sweep --config configs/sweeps/fusion.yaml
pvaug report --sweep runs/sweeps/fusion/sweep.json
```

评估输出准确率、宏平均 P/R/F1、加权 F1、逐类别指标、混淆矩阵和逐图概率。多种子报告统计已完成实验的均值与样本标准差。额外融合策略用于后续对照，不代表论文已经报告了这些结果。

## 代码位置

模型位于 `models/`，损失与学习任务位于 `tasks/`，训练循环和检查点位于 `engine/`。配置加载位于 `configuration/`，多阶段实验与参数网格位于 `orchestration/`。分类、扩散和 VQ 训练共用训练循环。

新增损失或模型的例子见 [architecture.md](architecture.md)。原始参数布局由兼容实现保留，历史源码放在 `reference/`。

## 原始权重与复现设置

`variant: paper` 使用 torchvision 骨干和公共 Sigmoid 门控融合；`variant: recovered` 保留早期分类网络，供原始权重加载。生成器兼容配置使用 L1 且未开启类别交叉注意力，论文配置使用 MSE 并开启类别条件，两者需要各自对应的权重。

```bash
pvaug convert-classifier --config configs/compatibility/classifier.yaml --checkpoint weights/legacy_classifier.pth --output weights/classifier.pt
pvaug convert-generator --config configs/compatibility/generator.yaml --checkpoint weights/legacy_generator.ckpt --output weights/generator.pt
```

其他实现差异见 [reproduction.md](reproduction.md)。当前验证覆盖 CPU 上的接口、损失、权重加载和恢复行为；真实数据精度与 GPU 性能尚未验证，具体记录见 [verification.md](verification.md)。

## 开发与上传

```bash
python -m ruff check src tests scripts
python -m pytest -q
python scripts/package_release.py --output ../pv-semantic-edge-github.zip
```

上传步骤见 [GitHub 上传说明](github_upload.md)。发布包排除了数据、模型权重、环境和运行目录。论文引用位于首页和 `CITATION.cff`，第三方来源见 `third_party/NOTICE.md`。
