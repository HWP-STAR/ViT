主要改进了 4 个方面，每个都直接针对之前不收敛的原因：
1. 换数据集：CIFAR-10（32×32 放大到 224）→ OxfordIIITPet（原生 224×224）
之前的问题：CIFAR-10 只有 32×32，resize 到 224×224 后每个 16×16 patch 只对应原始图像 ~2.3 个像素，patch 内全是插值噪声，Patch Embedding 提取不到有意义的信息。
为什么有效：OxfordIIITPet 原生就是 ~224×224 的自然图像，一个 16×16 patch 覆盖的是真实的结构化信息（毛发、眼睛、耳朵的形状），ViT 的 Self-Attention 才能学到跨 patch 的语义关联。
2. 数据增强：RandomResizedCrop + RandomHorizontalFlip
之前的问题：ViT 没有 CNN 的平移不变性和局部性归纳偏置（inductive bias），它把整张图切成 patch 后当序列处理。没有数据增强时，模型只能记住训练集的精确位置和方向，无法泛化。
为什么有效：RandomResizedCrop 模拟了不同缩放和裁剪（scale=0.8~1.0），相当于等效增大了训练数据量；HorizontalFlip 去掉了方向依赖。这些让模型学到"猫不管在图片的哪个位置、朝哪个方向，都是猫"。
3. Warmup + Cosine Annealing 学习率调度
之前的问题：固定 LR=1e-3，ViT 初期梯度方差大，大学习率直接导致梯度不稳定，模型根本不收敛（loss 停在 2.3 不动）。
为什么有效：
- Warmup（前 5 epoch 线性升温）：让学习率从 0 逐渐增加到设定值，ViT 在早期需要小学习率稳定训练，否则注意力权重的梯度在随机初始化时特别剧烈。
- Cosine Annealing（之后余弦退火到 0）：中后期 LR 平滑下降，避免在 loss landscape 的鞍点附近震荡，帮助精细收敛。
4. AdamW（weight_decay=0.05）替代 Adam
之前的问题：Adam 的 weight decay 实现方式有缺陷（等价于 L2 正则化但耦合了学习率），小模型容易过拟合噪声。
为什么有效：AdamW 把 weight decay 与学习率解耦，对大模型 ViT 是标准做法（DeiT/MAE 等 ViT 训练全部用 AdamW），0.05 的 weight decay 提供适度的正则化，防止对 37 类小数据集过拟合
