import torch 
import torch.nn as nn

VOCAB_SIZE=100
EMBED_DIM=4

embed=nn.Embedding(VOCAB_SIZE,EMBED_DIM)
input_ids=torch.tensor([1,5,9])

print(f'input_ids:{input_ids}')

output_v=embed(input_ids)

print("="*30)
print(f'output_v:{output_v}')


