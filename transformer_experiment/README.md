# transformer model 
1. data.py – 数据加载与预处理
主要功能：

WordTokenizer：构建词表、编码/解码文本。

TextDataset：将文本转换为模型输入（input_ids, labels）。

create_collate_fn：动态生成 collate 函数，避免全局 tokenizer。

load_wikitext_data：加载 WikiText-2 数据集并返回三个文本列表。

create_dataloaders：一键构建 DataLoader。

2. train.py – 训练、评估与生成
主要功能：

train_epoch：单轮训练，支持学习率调度器。

evaluate：计算验证集平均损失和困惑度。

generate：自回归文本生成。


3. utils.py – 辅助函数
主要功能：

设置随机种子（保证可复现）。

4. main.py – 主入口
整合所有模块，执行训练循环和交互式生成。
