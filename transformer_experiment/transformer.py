import torch
import torch.nn as nn
import torch.nn.functional as F
print('ok')

vocab_size=5000
embed_dim=256
num_heads=4
num_layers=4
max_seq_len=120
dropout=0.1

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1=nn.LayerNorm(embed_dim)
        self.attn=nn.MultiheadAttention(embed_dim,num_heads,dropout=dropout,batch_first=True)
        self.ln2=nn.LayerNorm(embed_dim)
        self.ffwd=nn.Sequential(
            nn.Linear(embed_dim,4*embed_dim),
            nn.GELU(),
            nn.Linear(4*embed_dim,embed_dim),
            nn.Dropout(dropout),
                )
    def forward(self,x):
        attn_out,_=self.attn(x,x,x,need_weights=False)
        x=x+attn_out
        x=self.ln1(x)
        ff_out=self.ffwd(x)
        x=self.ln2(x)
        return x

class MiniTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb=nn.Embedding(vocab_size,embed_dim)
        self.pos_emb=nn.Embedding(max_seq_len,embed_dim)
        self.blocks=nn.Sequential(*[TransformerBlock() for _ in range(num_layers)])
        self.ln_f=nn.LayerNorm(embed_dim)
        self.head=nn.Linear(embed_dim,vocab_size,bias=False)

    def forward(self,idx):
        B,T=idx.shape
        pos=torch.arange(0,T,device=idx.device).unsqueeze(0) #(1,T)
        
        x=self.tok_emb(idx)+self.pos_emb(pos)
        x=self.blocks(x)
        x=self.ln_f(x) #Norm lay
        logits=self.head(x) # Linear lay

        return logits

if __name__=="__main__":
    device=torch.device('cuda:0')
    print(f'using device:{device}')
    model=MiniTransformer().to(device)

    x=torch.randint(0,vocab_size,(2,64)).to(device)
    y=torch.randint(0,vocab_size,(2,64)).to(device)
    
    logits=model(x)
    loss = nn.CrossEntropyLoss()(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    print(f"测试通过，损失: {loss.item():.4f}")
