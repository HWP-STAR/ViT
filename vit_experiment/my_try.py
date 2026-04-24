import torch
import torch.nn as nn
import torch.nn.functional as F

#基础自注意力
class SingleHeadSelfAttention(nn.Module):
    def __init__(self,d_model):
        super().__init__()

        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)
        self.scale=torch.sqrt(torch.tensor(d_model,dtype=torch.float32))

    def forward(self,x):
        
        #QKV是向量
        Q=self.w_q(x)
        K=self.w_k(x)
        V=self.w_v(x)

        attn_scores=torch.matmul(Q,K.transpose(-1,-2)) / self.scale
        attn_weights=F.softmax(attn_scores,dim=-1)

        output=torch.matmul(attn_weights,V)
        return output,attn_weights

# 测试基础自注意力
d_model = 16  # 特征维度
seq_len = 5   # 序列长度
batch_size = 2
x = torch.randn(batch_size, seq_len, d_model)  # 模拟输入序列

s_h_attn=SingleHeadSelfAttention(d_model)
s_output,s_weights=s_h_attn(x)
print(f'output.shape:{s_output.shape}')
print(f'weight.shzpe;{s_weights.shape}')

#print(f'输出：{s_output}')

class MulHeadSelfAttention(nn.Module):
    def __init__(self,d_model,n_heads):
        super().__init__()
        assert d_model % n_heads == 0,"必须可以整除，否则会报错"

        self.n_heads=n_heads
        self.d_k=d_model // n_haeads #每个头的特征维度
        self.scale=torch.sqrt(torch.tensor(self.d_k,dtype=torch.float32))



