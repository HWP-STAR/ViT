import torch
import torch.nn as nn
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=4, num_layers=4, max_seq_len=128, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # 初始化参数
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, idx):
        B, T = idx.shape
        # 位置索引 (0 到 T-1)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)  # (1, T)
        # 词嵌入 + 位置嵌入
        x = self.token_embedding(idx) + self.position_embedding(pos)
        # 通过每个 Transformer Block
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)
        return logits

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffwd = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # 准备因果掩码 (上三角矩阵，禁止看到未来)
        B, T, C = x.shape
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        # 自注意力 (需要 mask)
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask)
        x = x + self.dropout(attn_out)
        x = self.ln1(x)
        ff_out = self.ffwd(x)
        x = x + self.dropout(ff_out)
        x = self.ln2(x)
        return x
