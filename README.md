# Vision Transformer (ViT) — 视觉分类模型

本项目实现了 ViT (Vision Transformer) 用于图像分类，基于 "An Image is Worth 16x16 Words" 论文。

## 模型参数对比：ViT vs 标准 Transformer 编码器-解码器

同样在 `embed_dim=384`、`n_head=6`、`num_layers=6` 配置下，ViT 的参数量（~8.7M）远少于标准 Transformer（~30M+），原因如下：

### 1. ViT 没有词嵌入层（Word Embedding）

标准 Transformer 需要两个巨大的词嵌入矩阵（源语言 + 目标语言词表 × d_model）。
例如 `vocab_size=10000`、`d_model=384`：
- 编码器嵌入: 10K × 384 = **3,840,000** 参数
- 解码器嵌入: 10K × 384 = **3,840,000** 参数

ViT 的 Patch Embedding 只是一个 Conv2d(3→384, kernel=16)：
- 3×384×16×16 + 384 = **295,296** 参数

**差距: ~7.68M vs 0.30M**

### 2. ViT 没有解码器（Decoder）

标准 Transformer 有编码器 + 解码器：
- **6 个解码器层** = 6 × (Self-Attention + Cross-Attention + FFN)
- 解码器每层比编码器多一个 Cross-Attention（4 个 Linear 映射），每层多出 ~0.59M 参数
- 6 层多出 ~3.54M 参数

ViT 只有编码器（12 个 Self-Attention + FFN 子层 vs 解码器的 18 个）。

### 3. ViT 的分类头很小

标准 Transformer 的输出层是词汇表投影：`Linear(d_model, vocab_size)` = 384×10K + 10K = **3,850,000** 参数。

ViT 的分类头：`Linear(d_model, num_classes)` = 384×37 + 37 = **14,245** 参数。

### 汇总对比

| 组件 | 标准 Transformer | ViT |
|---|---|---|
| 词嵌入 / Patch Embedding | ~7.68M | ~0.30M |
| 编码器 (6层) | ~8.29M | ~8.29M |
| 解码器 (6层, 含Cross-Attn) | ~11.83M | — |
| 输出投影 | ~3.85M | ~0.01M |
| **总计** | **~31.65M** | **~8.60M** |

结论：**ViT 只用编码器架构**，移除了词嵌入、解码器和词汇表投影，因此相同深度下参数量不到标准 Transformer 的三分之一。

## 数据集：Oxford-IIIT Pet

- 37 类猫狗品种
- 原生尺寸 ≈ 224×224（适合 ViT 的 16×16 patch）
- 训练集 trainval：~3,680 张
- 测试集 test：~3,669 张

## 训练要点

- Learning Rate Warmup（前 5 个 epoch 线性上升）+ Cosine Annealing 衰减
- 数据增强：RandomResizedCrop + RandomHorizontalFlip
- 优化器：AdamW (weight_decay=0.05)
- 每 100 batch 输出一次 loss / accuracy
- 自动保存最佳模型 `best_model.pth`
