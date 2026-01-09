from einops import rearrange
import torch

x=torch.randn(2,3,28,28)#(batch,channels,h,w)
print(f'x形状：（b,c,h,w）{x.shape}')

x1=rearrange(x,'b c h w -> b c (h w)')
print(f'x1形状 ；{x1.shape}')
