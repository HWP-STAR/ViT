import torch
import torch.nn as nn
from einops import rearrange

#FFN,全连接层
class FeedForward(nn.Module):
    def __init__(self,dim,mlp_dim=None,dropout=0.):
        super().__init__()

        mlp_dim=mlp_dim or dim *4
        self.net=nn.Sequential(
            nn.Linear(dim,mlp_dim),
            nn.GELU(),#激活函数
            nn.Dropout(dropout),
            nn.Linear(mlp_dim,dim),#先降低后升高
            nn.Dropout(dropout)
                )

    def forward(self,x):
        return self.net(x)

#多头自注意力模块
class MultiHeadAttention(nn.Module):
    def __init__(self,dim,heads=8,dropout=0.):
        super().__init__()
        self.heads=heads
        self.scale=(dim // heads)**-0.5
        inner_dim = dim 

        self.qkv=nn.Linear(dim,inner_dim *3,bias=False)
        self.proj=nn.Linear(inner_dim,dim)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        b,n,c=x.shape

        #生成QKV后拆分
        qkv=self.qkv(x).chunk(3,dim=-1)#拆分
        q,k,v=map(lambda t : rearrange(t,'b n (h d) -> b h n d', h=self.heads),qkv)
        #计算得分
        attn=(q@ k.transpose(-2,-1)) * self.scale
        attn = attn.softmax(dim=-1) #归一化
        attn=self.dropout(attn)

        #求和，投影对齐维度
        out=(attn @ v)
        out=rearrange(out,'b h n d -> b n ( h d)')
        out=self.proj(out) #投影
        out=self.dropout(out)

        return out

#transformer编码器
class TransformerBlock(nn.Module):
    def __init__(self,dim,heads=8,mlp_dim=None,dropout=0.):
        super().__init__()
        self.norm1=nn.LayerNorm(dim)
        self.attn=MultiHeadAttention(dim,heads,dropout)

        self.norm2=nn.LayerNorm(dim)
        self.ffn=FeedForward(dim,mlp_dim,dropout)

    def forward(self,x):
        x=x + self.attn(self.norm1(x)) #残差

        x= x + self.ffn(self.norm2(x)) #残差

        return x

#ViT模型
class ViT(nn.Module):
    def __init__(self,image_size=224,patch_size=16,num_classes=10,
            dim=768,#特征维数
            depth=2,#Transformer块数量
            heads=12,
            mlp_dim=None,#FFN隐藏层数
            channels=3,dropout=0.,emb_dropout=0.#嵌入层
            ):

            super().__init__()

            assert image_size % patch_size ==0 , "图像尺寸可以被分块整除"
            self.num_patches=(image_size// patch_size)**2 #总块数
            self.patch_size=patch_size
            patch_dim=channels*patch_size * patch_size #展平后的输入维数

            #1 图像块嵌入
            self.patch_embed=nn.Linear(patch_dim,dim)

            #2 类别
            self.cls_token=nn.Parameter(torch.randn(1,1,dim))

            #3 位置编码
            self.pos_embed=nn.Parameter(torch.randn(1,self.num_patches +1,dim))

            #4 dropout
            self.emb_dropout=nn.Dropout(emb_dropout)

            #5 Transformer堆叠
            self.transformer = nn.Sequential(*[
    TransformerBlock(dim,heads,mlp_dim,dropout)
    for _ in range(depth)
                ])

            #6 分类头
            self.norm=nn.LayerNorm(dim)
            self.head=nn.Linear(dim,num_classes)

    def forward(self,x):
        b,c,h,w = x.shape

        x=rearrange(x,'b c (h p1) ( w p2) -> b (h w) ( p1 p2 c)',
                p1=self.patch_size,p2=self.patch_size)

        #块嵌入
        x=self.patch_embed(x) # 展开，嵌入

        cls_tokens=self.cls_token.expand(b,-1,-1)
        x=torch.cat((cls_tokens,x),dim=1)

        # 添加位置编码 + dropout
        x = x + self.pos_embed
        x = self.emb_dropout(x)

        #编码器
        x=self.transformer(x)

        #提取第一个
        x=self.norm(x[:,0])
        
        return self.head(x)


    
