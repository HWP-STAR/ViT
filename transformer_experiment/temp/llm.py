from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
import torch
print("========start============")
model_name="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
import os
cache_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../data")

tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=cache_dir,local_files_only=True)

# 1. 配置 4bit 量化参数
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # 开启 4bit 量化加载
    bnb_4bit_quant_type="nf4",      # 使用 NF4 (Normalized Float 4) 格式，专为大模型权重设计，精度更高
    bnb_4bit_compute_dtype=torch.float16, # 计算时的数据类型，保持半精度以保证速度
    bnb_4bit_use_double_quant=True, # 开启双重量化，进一步节省显存
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map='auto',
    cache_dir=cache_dir ,local_files_only=True
)
print("============load_ok========")
'''
disk_path="../../data_disk/deepseeek-R1"
model.save_pretrained(disk_path)
tokenizer.save_pretrained(disk_path)
'''

device=torch.device("cuda:0")
print(f'using device:{device}')

prompt = "创作一篇内容深刻的短篇小说(800字左右,主题创新)"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# 3. 让模型生成回复
# max_new_tokens: 最多生成多少个新字/token
with torch.no_grad(): # 推理模式下关闭梯度计算，可以节省显存并加速
    outputs = model.generate(
        **inputs,
        max_new_tokens=1000,
        do_sample=True,      # 开启采样，让回答不那么死板
        temperature=0.7      # 温度越低越严谨，越高越有创意
    )

# 4. 解码并打印结果（skip_special_tokens=True 会过滤掉特殊的控制符）
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
