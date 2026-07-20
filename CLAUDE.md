# CLAUDE.md — 天文星图实例分割项目（MAINet）

> 本文件是项目的完整上下文记忆，供 Claude Code (CLI) 续接开发。
> 阅读后即可无缝接手，无需重新讨论已敲定的设计。

---

## 1. 项目目标

**任务**：地基望远镜星图的**实例分割**——输出每颗恒星的实例 mask（每颗星一个独立 mask，重叠区域可 overlap）。

**最终用途**：恒星提取。需要一个专门实验对比**双星检测**的 recall / mse 等指标。

**发表目标**：国内二三区 SCI/EI 期刊（不是顶会，组合创新即可，关键是实验完整 + 物理动机清晰）。

**时间**：1 个月。

---

## 2. 数据

### 2.1 数据形态
- **单帧**（无时序数据，这是和师兄 STAR 文章的关键区别，决定了天花板在二三区）
- 每幅图 **只有一种形态**：要么全是点源星，要么全是条状拖尾星，各占一半
- 图像尺寸 512×512，单通道灰度，存为 `.npy`

### 2.2 物理先验（极重要，是模型设计的核心依据）
**同一幅图内所有星的物理参数一致**：
- 点源星：σ_x, σ_y, θ（PSF 各向异性高斯，全图共享；仅用于形态仿真，**不输出监督**）
- 条状星：拖尾方向 φ、长度 L（全图共享）
- 星之间的区别只有：位置、亮度（flux）

### 2.3 三类目标
| 类别 | category_id | 训练权重 |
|---|---|---|
| single（普通星）| 1 | 1.0 |
| binary（双星，2σ mask 重叠）| 2 | 1.0 |
| faint（暗星）| 3 | SNR≥3→1.0, 1.5≤SNR<3→0.5, SNR<1.5→不标注 |

### 2.4 双星定义（已严格确定）
两颗星的 **2σ mask 有重叠**才算双星，不是质心距离。
- **点源双星**：质心间距 ∈ (PSF_SY×1.5, PSF_SY×4)，方向随机。太近无法区分，太远不重叠。
- **条状双星**：垂直拖尾方向偏移 perp ∈ (PSF_SY×0.3, PSF_SY×2.0) **且** 沿拖尾方向偏移 along ∈ (TRAIL_LENGTH×0.3, 0.8)。两条件同时满足。

### 2.5 标注格式
**COCO 格式**，实例级 id：
- 每颗星一个 annotation，有全数据集唯一的 `id`
- `segmentation` 是 RLE 编码的 2σ mask
- 重叠双星 = 两个独立 annotation，各自完整 mask，重叠区不裁剪（COCO 支持 overlap）
- annotation 额外字段：`centroid`, `flux`, `snr`, `weight`, `bbox`
- image 字段：`mode`(point/streak), `sigma_x`, `sigma_y`, `theta`, `phi`, `length`, `bg_noise_std`

### 2.6 数据生成与加速
- 生成脚本 `data/dataset_generator.py`（局部 bbox 计算加速，~1.1s/张）
- **点源用各向异性高斯 PSF**（σ_x≠σ_y，θ 随机），更真实；σ/θ 仅用于形态仿真 + mask 生成，**不参与训练监督**
- **训练加速关键**：mask 预解码为 npy。每张图合并存 `{id:06d}_masks.npy`([N,H,W]) + `{id:06d}_info.npy`([N,3]: ann_id,label,weight)
- 数据增强：随机水平/垂直翻转（点源翻转后 θ→-θ 同步；条状翻转后 φ→π-φ 或 -φ）
- 目录结构：`output/{train,val,test}/images/`, `output/{train,val,test}/masks/`, `output/annotations/{train,val,test}.json`

---

## 3. 模型架构：MAINet（Mask R-CNN 范式）

### 3.1 核心设计哲学
**用一套物理引导的机制，统一解决三个矛盾**（不是堆模块）：
1. 小目标 vs 下采样 → 全分辨率 Stem + 浅层不下采样
2. 点源 vs 条状（两种相反几何）→ 双路形态感知 + 门控融合
3. 双星 mask 重叠 → Mask R-CNN 实例分割（每 ROI 独立 mask，天然 overlap）

**与师兄 STAR 文章的关系**：STAR 用时序(TCE)做第二维度差异化；我们用**形态(点源/条状)**做第二维度。STAR 的 SCE 是普通大小核，我们的是**物理引导 PSF + 全局参数估计**，更高级。注意 PGFE 的大小核部分勿与 STAR 的 SCE 雷同，强调 PSF 物理分支为核心。

### 3.2 全局原则
**所有模块必须带残差连接。**（用户强制要求）

### 3.3 完整数据流（MAINetRCNN）
```
输入 [B,1,512,512]
  ↓
Stem (2×Conv3×3 + 原始信号残差) → [B,32,512,512]
  ↓
双路 Backbone:
  Stage1 DualBlock → [B,64,256,256]   (双路:PSF+条带)
  Stage2 DualBlock → [B,128,128,128]  (双路)
  Stage3 SingleBlock → [B,256,64,64]  (单路:深层拖尾已消失)
  Stage4 SingleBlock → [B,512,32,32]  (单路)
  ↓ 4个尺度特征
FPN → 4层融合特征 [B,256, H/s]
  ↓
RPN → proposals（区域候选框）
  ↓
ROIHead:
  BoxHead → 前景/背景分类 + bbox 回归
  MaskHead → 每个 ROI 独立 mask [28×28] → 上采样到原图
  ↓
N个实例 mask，重叠区天然 overlap
```

### 3.4 双路 DualBlock 内部（Stage1-2，核心创新）
每个 DualBlock 包含：

**① 全局物理参数估计 (GlobalParamEstimator)**
- 1×1降维 → 池化到32×32 → 轻量自注意力 → 全局池化
- 输出全图共享参数：σ_x,σ_y,θ (PSF) + φ,L (拖尾) + gate_logit(门控分类)
- 物理激活：σ用softplus+0.5(>0)，θ/φ用tanh×(π/2)，L用softplus+1
- 注意力前必须先池化到32×32，否则512²注意力爆显存

**② PSF 通道 (PSFChannel)**
- 用全局估计的(σx,σy,θ)**动态生成**各向异性Gaussian核 [B,1,7,7]（k=7覆盖2σ）
- depthwise（全通道共享一个PSF核），fold batch到通道做 grouped conv
- 1×1卷积 + 残差

**③ 条带通道 (StripChannel)**
- 旋转特征图（affine_grid + grid_sample，可微，0填充）
- 十字长条卷积（depthwise）：水平 1×15（沿拖尾）+ 垂直 5×1（跨拖尾），**相加**融合
- 旋转回原坐标系
- 1×1卷积 + 残差

**④ 门控融合 (GatedFusion)**
- 标量 gate [B,1] = Sigmoid(gate_logit)
- 软融合：fused = gate×PSF特征 + (1-gate)×条带特征
- 残差用 Block 原始输入：out = identity + fused
- **gate 训练监督**：point 图 target=0，streak 图 target=1（BCEWithLogits）

**⑤ 融合后下采样**（先双路融合，再下采样，顺序已确认）

### 3.5 后半段（Mask R-CNN 范式，成熟方案）
- **FPN**：4尺度自顶向下融合，输出 256 通道特征
- **RPN**：AnchorGenerator + RPNHead，生成区域候选框
- **ROIHead**：
  - BoxHead：fc→1024→cls(前/背2类)+reg(4坐标)
  - MaskHead：4层Conv→1×1Conv，输出 [N,1,28,28] mask

### 3.6 v3 Backbone（零参数标注，新增）
**morph_modules_v3.py** 提供全新的点源/条带/融合模块，**完全不需要物理参数估计**：
- **MoffatPSFChannel**：可学习 Moffat 剖面 (1+r²/α²)^(-β) 多尺度核 + 像素级尺度注意力。beta 控制翼区厚度，幂律衰减比高斯更贴合真实星点外围。K 个尺度并行覆盖星点大小差异。
- **MultiOrientStripChannel**：固定 4 方向 (0/45/90/135) 旋转 + 长条卷积 + 注意力选向。无需估计/监督 φ，天然覆盖任意拖尾方向。
- **SKFusion**：通道维 + 空间维双重选择性融合（SKNet 风格），从特征本身算门控，不依赖 estimator。
- **DualBlockV3**：MoffatPSF + MultiOrientStrip + SKFusion 三件套组装，无 params 输入、无 estimator。
- **dual_path_v3.py**：Stem → 2×DualBlockV3 → 2×SingleBlock（与 v1/v2 相同后半段）。
- **关键创新**：零标注需求 — 不需要 σ/θ/φ/L 估计，不需要 param_loss，完全靠可学习先验自适应。

### 3.7 类别处理（关键）
**类别不参与训练**：训练时所有星视为单一前景类（num_classes=1，只分前景/背景）。
每个实例的 single/binary/faint 标签保留在标注里。**推理后**根据匹配上的GT给预测实例赋类别，用于双星专项评估。

---

## 4. 损失与训练

### 4.1 RCNN 损失（RCNNCriterion）
- **RPN**：cls(前/背二分类) + reg(smooth_l1 bbox回归)
- **ROI**：cls(前/背二分类) + reg(bbox回归) + mask(BCE, SNR加权)
- SNR加权：暗星 weight=0.5（从GT附带的weight字段）

### 4.2 物理参数辅助监督（仅 v2 backbone，param_loss.py）
- **收窄版监督**：只监督 φ（方向）和 gate（点/线门控分类）
- σ/θ/L **不再强监督**（避免过拟合合成假设）
- φ 监督：sin(2φ)/cos(2φ) smooth_l1（仅 streak 图）
- gate 监督：BCEWithLogits，mode='streak'→1, mode='point'→0（所有图）
- 权重从0.1线性衰减到0（前期帮收敛，后期放开）
- **消融实验**：有参数监督 vs 无

### 4.3 Backbone 版本
| --model | Backbone | param 监督 |
|---------|----------|-----------|
| rcnn_v1 | DualPath v1（每 block 自带 estimator） | 无 |
| rcnn_v2 | DualPath v2（顶层 estimator + param 监督） | φ + gate |
| rcnn_v3 | DualPath v3（MoffatPSF + MultiOrientStrip + SKFusion，零标注） | 无 |
| rcnn_resnet | ResNet-18（消融 baseline） | 无 |

### 4.4 论文只用合成数据还是真实数据验证
- 用户担心辅助监督导致过拟合合成假设、真实数据表现差
- 决策：收窄版监督（仅φ+gate）+ 消融实验。若真要真实数据验证，需更谨慎

---

## 5. 当前进度

### 目录结构
```
MY_query_mask/                    ← 项目根（工作目录）
├── train.py                      ← 训练入口：python train.py --model v3 --debug --epochs 1
├── run_all.py                    ← 一键训练所有模型
├── eval.py                       ← 统一评估 work_dirs/ 下所有模型
├── visualize.py                  ← 交互式可视化
├── plot_ap.py                    ← 多模型 PR 曲线对比
│
├── models/                       ← ★ 每个模型自包含完整训练管线
│   ├── mainet/
│   │   ├── v1/                   ← DualPath v1（每 block 自带 estimator）
│   │   │   ├── train.py          ← 训练脚本（Config + 训练循环 + CLI）
│   │   │   ├── model.py          ← MAINetRCNN + RCNNCriterion
│   │   │   ├── backbone.py       ← DualPathBackbone v1
│   │   │   ├── heads.py          ← FPN + AnchorGenerator + RPN + ROIHead
│   │   │   └── dataset.py        ← Dataset + collate + asinh 归一化
│   │   ├── v2/                   ← DualPath v2 + φ/gate 参数监督
│   │   │   ├── ...同上 + param_loss.py
│   │   ├── v3/                   ← DualPath v3（MoffatPSF + MultiOrientStrip + SKFusion）
│   │   ├── resnet/               ← ResNet-18 消融 baseline
│   │   └── query/                ← Query-based MAINet（DETR 风格）
│   └── mmdet/                    ← mmdet 对比模型（薄封装，依赖 mmdet 库）
│       ├── mask_rcnn/train.py
│       ├── solov2/train.py
│       ├── condinst/train.py
│       └── mask2former/train.py
│
├── mainet/                       ← 共享模型库（eval/visualize 兼容用，新训练不用）
│   ├── backbones/                ← 原始 backbone 代码
│   ├── rcnn_head/                ← 原始 FPN/RPN/ROIHead
│   ├── query_head/               ← 原始 Query 模型
│   ├── param_loss.py
│   └── train.py                  ← 旧入口（DEPRECATED，转发到 models/）
│
├── data/                         ← 共享数据管线
│   ├── dataset_generator.py      ← 模拟数据生成
│   ├── dataset.py                ← PyTorch Dataset
│   └── normalize.py              ← asinh 归一化
│
├── mmdet/                        ← mmdet 配置 + 训练逻辑
│   ├── configs/                  ← *_star.py 配置文件
│   ├── train_compare.py          ← mmdet 训练入口
│   └── npy_loading.py            ← npy 图像加载
│
├── output/                       ← 共享数据
│   └── {train,val,test}/{images,masks}/ + annotations/
│
└── work_dirs/                    ← ★ 所有模型权重统一输出
    ├── mainet/v1/best_model.pt
    ├── mainet/v2/best_model.pt
    ├── mainet/v3/best_model.pt
    ├── mmdet/mask_rcnn/
    └── ...（同时兼容旧扁平结构 mainet_*）
```

### 关键用法
```bash
# 训练（从项目根目录）
python train.py --model v3 --debug --epochs 1       # 单模型冒烟测试
python train.py --model v2 --epochs 100              # 全量训练
python run_all.py --epochs 100                       # 一键训练所有
python run_all.py --only v2 v3 --dry-run             # 预览

# 评估 / 可视化
python eval.py                                        # 自动扫描 work_dirs/
python visualize.py --ckpt work_dirs/mainet/v3        # 交互式浏览
python plot_ap.py                                     # PR 曲线对比

# 新增模型：复制任意文件夹，修改 Config 和 backbone.py
cp -r models/mainet/v3 models/mainet/v4
```

### 设计原则
- **每个 models/mainet/<name>/ 是完整独立项目**：train.py + model.py + backbone.py + heads.py + dataset.py
- **复制到另一台电脑即可训练**：只需 torch/numpy/tqdm，零项目内部依赖
- **所有 checkpoint 统一在 work_dirs/**，按框架/模型名两级目录
- **eval.py / visualize.py 自动兼容新旧两种 work_dirs 结构**

---

## 6. 关键技术注意点

1. **AMP dtype**：RCNN 动态 mask head 卷积在 AMP 下 half/float 可能不匹配，loss 内部转 float32 防溢出。
2. **PSF 动态核 batch 处理**：每张图核不同，用 fold batch 到通道 + groups=B*C 的 grouped conv。
3. **条带旋转**：affine_grid 旋转矩阵每张图角度不同，逐图构造 [B,2,3]。0填充。
4. **自注意力显存**：必须先池化到 32×32 再做注意力，512² 直接做会爆。
5. **角度周期性**：φ 的监督用 sin(2φ)/cos(2φ)，不能直接回归角度值（0 和 π 等价）。
6. **数据加载瓶颈**：必须用预解码的合并 npy（见 2.6），否则 RLE decode 拖慢训练。
7. **NaN 检测**：mmdet train_compare 有 NaNStopperHook 即时停止；mainet train.py 有 grad_clip=1.0。
8. **类别不参与训练**：num_classes=1（前景/背景），single/binary/faint 推理后匹配赋值。
9. **v3 零标注设计**：MoffatPSF 可学习 alpha/beta 替代固定 σ，MultiOrientStrip 固定 4 方向替代 φ 估计，SKFusion 从特征算 gate 无需 estimator。v3 训练时 params 参数传 None，RCNNCriterion 自动跳过 param 相关 loss。

---
## 7. 命名

- 网络：MAINet (Morphology-Aware Instance Network)
- 模块（v1/v2）：PSFChannel / StripChannel / GatedFusion / GlobalParamEstimator / DualBlock
- 模块（v3）：MoffatPSFChannel / MultiOrientStripChannel / SKFusion / DualBlockV3
- 论文创新点：
  1. 全局物理参数引导的双路形态感知（PSF 点源路径 + 方向条带路径 + 门控融合）— v1/v2
  2. 旋转条带卷积处理拖尾的方向性长程结构 — v1/v2
  3. 可学习 Moffat 多尺度核 + 多方向固定角度条带 + SK 选择性融合（零参数标注）— v3
  3. 形态感知 Backbone + Mask R-CNN 框架处理重叠双星 + SNR 加权 + 合成数据策略

---

## 8. 用户偏好

- 要求所有模块带残差
- 倾向自己掌控核心创新（Backbone），后半段用成熟方案（Mask R-CNN）
- 担心过拟合合成数据 → 重视泛化、用消融实验佐证
- σ/θ 参数仅用于形态仿真不监督
- 沟通风格：喜欢先讲清设计逻辑再写代码，会质疑不合理的设计
