import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import MultiHeadAttention, PositionwiseFeedForward, EncoderLayer

# ====== ViT model ======
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 num_classes=100, embed_dim=768, n_head=12, num_layers=12, d_ff=3072,
                 dropout=0.1, device='cuda:0'):
        super().__init__()
        self.device = device
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.n_patches = (img_size // patch_size) ** 2

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(p=dropout)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(embed_dim, n_head, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        for layer in self.encoder_layers:
            x = layer(x, mask=None)
        x = self.norm(x)

        cls_out = x[:, 0]
        logits = self.head(cls_out)
        return logits


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'using device: {device}')

    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=100,
        embed_dim=384,
        n_head=6,
        num_layers=6,
        d_ff=1024,
        dropout=0.1,
        device=device
    ).to(device)

    images = torch.randn(2, 3, 224, 224).to(device)
    logits = model(images)
    print('=' * 50)
    print(f'Input shape: {images.shape}')
    print(f'Logits shape: {logits.shape}')

    labels = torch.randint(0, 100, (2,)).to(device)
    print(f'labels shape:{labels.shape}')
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, labels)
    print(f"分类损失: {loss.item():.4f}")






