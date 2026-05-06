from transformers import AutoModelForCauslLM,AutoTokenizer
import torch
print('ok')

device=torch.device("cuda:0")
print(f'using device:{device}')
disk="../../data_disk/SmolLM-135M/"


