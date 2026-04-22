import torch
import torch.nn as nn
from einops import rearrange

# ================================
# 1. 多头自注意力模块 (简化版+最优实现)
# ================================
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5  # 缩放因子，避免点积过大导致梯度消失
        inner_dim = dim  # 保持总维度与输入一致，简化计算
        
        # 合并QKV投影层，减少网络层声明，提升效率
        self.qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.proj = nn.Linear(inner_dim, dim)  # 多头输出合并后的投影层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, c = x.shape  # b:批次, n:序列长度(图像块数), c:特征维度
        
        # 生成QKV并拆分 (b, n, 3*c) -> 3*(b, n, c)
        qkv = self.qkv(x).chunk(3, dim=-1)
        # 重塑为多头格式 (b, heads, n, c//heads)，实现多头并行计算
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        # 计算注意力分数 + softmax归一化
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # 注意力加权求和 + 多头合并 + 最终投影
        out = (attn @ v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)
        out = self.dropout(out)
        
        return out

# ================================
# 2. 前馈神经网络 (FFN) - ViT标配结构
# ================================
class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim=None, dropout=0.):
        super().__init__()
        mlp_dim = mlp_dim or dim * 4  # 默认隐藏层维度为输入4倍，ViT官方推荐值
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),  # ViT专属激活函数，效果优于ReLU
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# ================================
# 3. Transformer编码器块 - Pre-Norm结构 训练更稳定
# ================================
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_dim=None, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)  # 注意力层前归一化
        self.attn = MultiHeadAttention(dim, heads, dropout)
        
        self.norm2 = nn.LayerNorm(dim)  # 前馈层前归一化
        self.ffn = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x):
        # 残差连接 + 自注意力
        x = x + self.attn(self.norm1(x))
        # 残差连接 + 前馈网络
        x = x + self.ffn(self.norm2(x))
        return x

# ================================
# 4. 完整ViT模型 (核心简化版+修复BUG+可直接运行)
# ================================
class ViT(nn.Module):
    def __init__(
        self,
        image_size=224,    # 输入图像尺寸
        patch_size=16,     # 图像分块大小
        num_classes=1000,  # 分类类别数
        dim=768,           # 特征维度
        depth=12,          # Transformer块堆叠数量
        heads=12,          # 注意力头数
        mlp_dim=None,      # FFN隐藏层维度
        channels=3,        # 图像通道数，RGB图=3，灰度图=1
        dropout=0.,        # 通用dropout概率
        emb_dropout=0.     # 嵌入层dropout概率，防止过拟合
    ):
        super().__init__()
        
        # 强制校验：图像尺寸必须能被分块大小整除，否则无法均分图像
        assert image_size % patch_size == 0, "图像尺寸必须能被分块大小整除！"
        self.num_patches = (image_size // patch_size) ** 2  # 计算图像被切分后的总块数
        patch_dim = channels * patch_size * patch_size      # 单块图像展平后的维度
        
        # ========== 修复点1：保存patch_size为类的成员属性 ==========
        self.patch_size = patch_size
        
        # 图像块嵌入层：将展平的图像块 线性投影到指定特征维度dim
        self.patch_embed = nn.Linear(patch_dim, dim)
        
        # Class Token：可学习的全局特征聚合向量，ViT核心组件
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # 位置编码：可学习的位置信息，弥补Transformer无顺序的缺陷
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        
        # 嵌入层dropout
        self.emb_dropout = nn.Dropout(emb_dropout)
        
        # Transformer编码器堆叠，用Sequential简化循环调用
        self.transformer = nn.Sequential(*[
            TransformerBlock(dim, heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        
        # 最终分类头
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        b, c, h, w = x.shape  # 输入张量形状: (批次, 通道, 高, 宽)
        
        # ========== 修复点2：调用self.patch_size，而不是直接用patch_size ==========
        # 图像分块+展平：(b, c, h, w) -> (b, num_patches, patch_dim)
        x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        
        # 图像块线性投影
        x = self.patch_embed(x)
        
        # 添加Class Token：扩展到批次维度 + 拼接到图像块序列最前面
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 添加位置编码 + dropout正则化
        x = x + self.pos_embed
        x = self.emb_dropout(x)
        
        # 经过Transformer编码器提取特征
        x = self.transformer(x)
        
        # 提取Class Token的输出做分类（ViT的核心分类逻辑）
        x = self.norm(x[:, 0])
        
        # 最终分类输出
        return self.head(x)

# ================================
# 5. 测试代码 (验证模型无报错+可正常运行)
# ================================
if __name__ == "__main__":
    # 轻量化测试参数，适合CPU快速运行，无硬件要求
    image_size = 32
    patch_size = 4
    num_classes = 10
    dim = 128
    depth = 2
    heads = 4
    mlp_dim = 256
    
    # 初始化模型
    model = ViT(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        dropout=0.1,
        emb_dropout=0.1
    )
    
    # 创建随机测试输入 (batch_size=2, channel=3, H=32, W=32)
    test_input = torch.randn(2, 3, image_size, image_size)
    
    # 前向传播测试，禁用梯度计算加速
    with torch.no_grad():
        output = model(test_input)
    
    # 打印结果验证
    print("="*60)
    print("✅ 模型运行成功！无报错")
    print(f"输入张量形状: {test_input.shape}")
    print(f"输出张量形状: {output.shape} (预期: [2, {num_classes}])")
    print(f"模型总参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("="*60)
