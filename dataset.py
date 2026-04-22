from datasets import load_dataset
print("ok")
"""
# 1. 加载一个非常轻量级的数据集
# 对于语言建模，'wikitext-2' 是经典的入门选择
dataset = load_dataset(
        'wikitext', 'wikitext-2-raw-v1',
        cache_dir="../data"
                        
                       )
print(dataset)  # 看看数据分成训练集、验证集和测试集

# 2. 简单看一下数据长什么样
# 比如打印训练集的前2条样本
print(dataset['train'][:2])
print("over")

"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
# 加载一个小型预训练模型和它对应的分词器
model_name = "HuggingFaceTB/SmolLM-135M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             cache_dir="../data")

# 确保模型在笔记本上能高效运行
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f'using device:{device}')

# 测试一下预训练模型
input_text = "The meaning of life is"
inputs = tokenizer(input_text, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
