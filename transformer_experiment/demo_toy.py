# 构造一个“数字加 1”的玩具数据集
from models import MiniTransformer
import torch
import torch.nn as nn
# 生成数据函数
def generate_plus_one_data(num_samples=100, seq_len=5, start_min=1, start_max=50):
    X = []
    Y = []
    for _ in range(num_samples):
        start = torch.randint(start_min, start_max, (1,)).item()
        seq = torch.arange(start, start + seq_len)
        X.append(seq[:-1])   # 输入: 前 seq_len-1 个数字
        Y.append(seq[1:])    # 输出: 后 seq_len-1 个数字
    return torch.stack(X), torch.stack(Y)

# 词汇表大小需要覆盖所有数字 + 特殊token（比如我们只用数字，vocab_size=100足够）
vocab_size = 100
seq_len = 5
train_X, train_Y = generate_plus_one_data(num_samples=200, seq_len=seq_len, start_min=1, start_max=vocab_size-seq_len)

device=torch.device("cuda:0")
print(f'using device:{device}')

model=MiniTransformer(vocab_size=vocab_size,embed_dim=64,num_heads=2,num_layers=2,max_seq_len=seq_len).to(device)
opt=torch.optim.Adam(model.parameters(),lr=1e-3)
train_X, train_Y = train_X.to(device), train_Y.to(device)

for epoch in range(200):
    logits=model(train_X)

    loss = nn.CrossEntropyLoss()(logits.view(-1, vocab_size), train_Y.view(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()
    if epoch % 40 ==0:
        print(f'Epoch{epoch},loss:{loss.item():.4f}')
