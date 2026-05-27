from datasets import load_dataset,load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
print("ok")

# 加载一个小型预训练模型和它对应的分词器
model_name = "HuggingFaceTB/SmolLM-135M"

tokenizer = AutoTokenizer.from_pretrained(model_name,
        local_files_only=True)

device=torch.device("cuda:0")
model = AutoModelForCausalLM.from_pretrained(
        model_name,cache_dir="../../data",
        local_files_only=True
        #device_map="cuda:0",
        #low_cpu_usage=True,
        #torch_dtype=torch.float16
                ).to(device)

# 确保模型在笔记本上能高效运行
print(f'using device:{device}')

if __name__=="__main__":
        input_text = "tell me a fun story,long time ago,I see a man .I asked him where is he coming from"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=100,temperature=0.7)
        print(tokenizer.decode(outputs[0]))

        帮我修改main_smollm.py(我想使用该模型smollm来训练,观察loss变化),完成后告诉我(不用运行完整程序,没有报错就可以)