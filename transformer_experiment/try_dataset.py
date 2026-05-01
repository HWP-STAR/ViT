from datasets import load_dataset,load_from_disk
print("ok")

# 1. 加载一个非常轻量级的数据集
# 对于语言建模，'wikitext-2' 是经典的入门选择
'''
dataset = load_dataset(
        'wikitext', 'wikitext-2-raw-v1',
        cache_dir="../../data")
#print(dataset)  # 看看数据分成训练集、验证集和测试集
#dataset.save_to_disk("../../data")
'''

dataset=load_from_disk("../../data")
print(f'dataset:\n{dataset}')
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

device=torch.device("cuda:0")
model = AutoModelForCausalLM.from_pretrained(
        model_name,cache_dir="../../data",
        #device_map="cuda:0",
        #low_cpu_usage=True,
        #torch_dtype=torch.float16
                ).to(device)

# 确保模型在笔记本上能高效运行
print(f'using device:{device}')

# 测试一下预训练模型
while True:
    input_text=input("input text(q to exit):")
    if input_text == 'q':
        print('over')
        break
    else:

        #input_text = "tell me a fun story,long time ago"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=100,temperature=0.7)
        print(tokenizer.decode(outputs[0]))

"""
