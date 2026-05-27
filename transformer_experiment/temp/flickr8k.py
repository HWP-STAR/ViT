from datasets import load_dataset

data_name="shivangibithel/Flickr8k"
cache_dir="../../../data"
# 自动下载并缓存（约 1.1GB）
dataset = load_dataset("jxie/flickr8k",
                       cache_dir=cache_dir)

print('='*30)

# 查看结构
print(dataset)
# 取训练集第一个样本
sample = dataset["train"][0]

# 显示图片
sample["image"].show()

# 打印 5 个 caption
for i in range(5):
    print(f"caption_{i}:", sample[f"caption_{i}"])
