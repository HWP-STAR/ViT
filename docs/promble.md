模型完全没有学习，和随机猜测一致。根本原因有以下几点：
1. CIFAR-10 原生 32×32 → 强行 resize 到 224×224
这是最致命的问题。CIFAR-10 原始图像只有 32×32 像素，resize 到 224×224 后严重模糊。你的 patch_size=16，意味着 224×224 中每个 16×16 的 patch 只对应原始图像的 ~2.3×2.3 个像素，Patch Embedding 几乎提取不到有意义的信息：
# ViT 的 patch_embed 是把每个 16x16 区域压成一个 token
# 原图 32x32 → 放大到 224x224 → 每个 patch 只覆盖 ~2 个原始像素
# 这等价于在说 "用这 3 个像素点猜这是飞机还是汽车"
2. 缺少 Learning Rate Warmup
ViT 对优化非常敏感，从零训练时必须用 warmup（典型：线性 warmup 500~1000 steps），否则早期梯度不稳定导致模型根本不收敛。
3. 10 个 epoch 远远不够
ViT 从零训练小数据集通常需要 100~300+ epochs（参考 DeiT 在 ImageNet 上训练 300 epochs），10 个 epoch 连热身都不够。
4. 缺少数据增强
CIFAR-10 训练标配：RandomCrop + RandomHorizontalFlip + CutMix/MixUp。ViT 没有 CNN 的归纳偏置（locality, translation invariance），需要更多数据增强来弥
