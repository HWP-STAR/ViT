import torch
import torch.nn as nn
import torch.nn.functional as F
print('ok')

class SingleHeadAttention(nn.Module):
    def __init__(self,d_model,d_k):
        """
        d_model: 输入特征维度
        d_k    : Q/K 的投影维度 (通常 d_v 也等于 d_k)
        """
        super().__init__()
        self.W_Q=nn.Linear(d_model,d_k,bias=False)
        self.W_K=nn.Linear(d_model,d_k,bias=False)
        self.W_V=nn.Linear(d_model,d_k,bias=False)

    def forward(self,x):
        Q=self.W_Q(x)
        K=self.W_K(x)
        V=self.W_V(x)

        # 计算注意力分数: Q * K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.W_Q.out_features ** 0.5)
        attn_weights=F.softmax(scores,dim=-1)

        output=torch.matmul(attn_weights,V)
        return output,attn_weights
class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,h):
        super().__init__()

        assert d_model%h ==0,'not 0'
        self.d_k=d_model//h
        self.h=h

        self.W_Q=nn.Linear(d_model,d_model,bias=False)
        self.W_K=nn.Linear(d_model,d_model,bias=False)
        self.W_V=nn.Linear(d_model,d_model,bias=False)
        self.W_O=nn.Linear(d_model,d_model,bias=False)

    def forward(self,x):
        batch,seq_len,_=x.shape

        Q=self.W_Q(x)
        K=self.W_K(x)
        V=self.W_V(x)


        # 2. 拆分成多个头: (batch, seq_len, h, d_k) -> 转置为 (batch, h, seq_len, d_k)
        Q = Q.view(batch, seq_len, self.h, self.d_k).transpose(1, 2)
        K = K.view(batch, seq_len, self.h, self.d_k).transpose(1, 2)
        V = V.view(batch, seq_len, self.h, self.d_k).transpose(1, 2)

        scores=torch.matmul(Q,K.transpose(-2,-1))/self.d_k **0.5

        attn_weights=F.softmax(scores,dim=-1)
        head_output=torch.matmul(attn_weights,V)

        #mix
        head_output=head_output.transpose(1,2).contiguous().view(batch,seq_len,-1)

        output=self.W_O(head_output)
        return output,attn_weights


if __name__=="__main__":
    batch=2
    seq_len=3
    d_model=8

    x=torch.randn(batch,seq_len,d_model)

    single=SingleHeadAttention(d_model,d_k=8)
    out_single,attn_single=single(x)
    print(f'out_single:\n{out_single} \natt_single:\n{attn_single}')

    # 多头注意力 (4 个头，每个头维度 8//4=2)
    multi = MultiHeadAttention(d_model, h=4)
    out_multi, attn_multi = multi(x)
    print("多头注意力输出形状:", out_multi.shape)       # (2,3,8)
    print("多头注意力权重形状:", attn_multi.shape)      # (2,4,3,3)

    # 对比: 单头权重是 (batch, seq, seq) ，多头权重多了头维度 (batch, h, seq, seq)
    print("\n单头权重示例 (第一个样本):\n", attn_single[0])
    print("\n多头第一个头的权重示例:\n", attn_multi[0, 0])
