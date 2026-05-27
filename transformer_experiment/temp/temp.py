from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,MarianTokenizer
from datasets import load_dataset

print('ok')
# 指定模型名称
model_name = "Helsinki-NLP/opus-mt-en-de"

# 加载分词器 (Tokenizer) 和模型 (Model)
# 首次运行时会自动下载模型文件
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name,
                                              cache_dir="../../data/")

model.save_pretrained("../../data_disk/opus-mt-en-de")
tokenizer.save_pretrained("../../data_disk/opus-mt-en-de")
# 官方原始版本可能需要特定的配置，使用封装好的版本更稳定
dataset = load_dataset("bentrevett/multi30k")

# 查看数据集结构
print(dataset)

dataset.save_to_disk("../../data_disk/multi30k")
print('over')



