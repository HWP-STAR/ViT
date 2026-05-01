import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from collections import Counter
import math
import random
import numpy as np

class WordTokenizer:
    def __init__(self, vocab_size=5000, unk_token="<UNK>", pad_token="<PAD>"):
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.word2idx = {}
        self.idx2word = {}
    
    def build_vocab(self, texts):
        """texts: list of strings"""
        # 简单的分词：按空格分割，保留原始标点（不额外处理）
        word_counts = Counter()
        for text in texts:
            words = text.strip().split()
            word_counts.update(words)
        
        # 保留最常见的 vocab_size-2 个词（留给 UNK 和 PAD）
        most_common = word_counts.most_common(self.vocab_size - 2)
        
        # 添加特殊token
        self.word2idx[self.pad_token] = 0
        self.word2idx[self.unk_token] = 1
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        print(f"Vocabulary built: {len(self.word2idx)} words (including PAD/UNK)")
    
    def encode(self, text, max_len=None):
        """将文本转换为索引列表"""
        words = text.strip().split()
        indices = [self.word2idx.get(word, self.word2idx[self.unk_token]) for word in words]
        if max_len is not None:
            if len(indices) > max_len:
                indices = indices[:max_len]
            else:
                indices = indices + [self.word2idx[self.pad_token]] * (max_len - len(indices))
        return indices
    
    def decode(self, indices):
        """将索引列表转换回文本"""
        words = [self.idx2word.get(idx, self.unk_token) for idx in indices]
        return " ".join(words)
    
    def vocab_size(self):
        return len(self.word2idx)


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # 编码并截断到 max_len
        ids = self.tokenizer.encode(text, max_len=self.max_len)
        # 对于语言模型，输入和标签相同（预测下一个词）
        input_ids = torch.tensor(ids[:-1], dtype=torch.long) if len(ids) > 1 else torch.tensor([], dtype=torch.long)
        labels = torch.tensor(ids[1:], dtype=torch.long) if len(ids) > 1 else torch.tensor([], dtype=torch.long)
        return input_ids, labels

def collate_fn(batch):
    # batch: list of (input_ids, labels)
    # 找出本 batch 中最长的序列长度
    max_len = max([inp.size(0) for inp, _ in batch if inp.size(0) > 0])
    if max_len == 0:
        # 如果所有样本都为空，返回空 batch
        return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)

    padded_inputs = []
    padded_labels = []
    for inp, lbl in batch:
        if inp.size(0) == 0:
            continue
        # pad 到 max_len
        pad_len = max_len - inp.size(0)
        inp_padded = F.pad(inp, (0, pad_len), value=tokenizer.word2idx[tokenizer.pad_token])
        lbl_padded = F.pad(lbl, (0, pad_len), value=-100)  # -100 会被交叉熵忽略
        padded_inputs.append(inp_padded)
        padded_labels.append(lbl_padded)
    return torch.stack(padded_inputs), torch.stack(padded_labels)

dataset = load_dataset(
        'wikitext', 'wikitext-2-raw-v1',
        cache_dir="../../data",
            trust_remote_code=False,
download_mode="reuse_cache_if_exists")

train_texts = [ex['text'] for ex in dataset['train'] if ex['text'].strip()]
valid_texts = [ex['text'] for ex in dataset['validation'] if ex['text'].strip()]
test_texts  = [ex['text'] for ex in dataset['test'] if ex['text'].strip()]

print(f"Train samples: {len(train_texts)}, Valid: {len(valid_texts)}, Test: {len(test_texts)}")

# 构建词表（只用训练集）
tokenizer = WordTokenizer(vocab_size=5000)
tokenizer.build_vocab(train_texts)

VOCAB_SIZE = tokenizer.vocab_size
print(f"Final vocab size: {VOCAB_SIZE}")


# 创建 Dataset 和 DataLoader
train_dataset = TextDataset(train_texts, tokenizer, max_len=128)
valid_dataset = TextDataset(valid_texts, tokenizer, max_len=128)

BATCH_SIZE =64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

print('data loader is ok')

def train_epoch(model, dataloader, optimizer, scheduler=None):
    model.train()
    total_loss = 0
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        if inputs.size(0) == 0:
            continue
        optimizer.zero_grad()
        logits = model(inputs)  # (B, T, vocab_size)
        # 计算损失 (忽略 -100 标签)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), labels.view(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        total_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}, loss: {loss.item():.4f}")
    return total_loss / len(dataloader)

def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            if inputs.size(0) == 0:
                continue
            logits = model(inputs)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), labels.view(-1), ignore_index=-100, reduction='sum')
            total_loss += loss.item()
            total_tokens += (labels != -100).sum().item()
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    return avg_loss, perplexity
def generate(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0):
    model.eval()
    input_ids = tokenizer.encode(prompt, max_len=MAX_SEQ_LEN)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids)  # (1, T, vocab_size)
        # 只取最后一个位置的 logits
        next_token_logits = logits[0, -1, :] / temperature
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        if next_token == tokenizer.word2idx[tokenizer.pad_token]:
            break
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
        if input_ids.size(1) >= MAX_SEQ_LEN:
            break
    generated = tokenizer.decode(input_ids[0].tolist())
    return generated


if __name__=="__main__":
    from try_model import MiniTransformer

    print('start tain')
    print('=='*30)

    EMBED_DIM=512
    NUM_HEADS=4
    NUM_LAYERS=16
    MAX_SEQ_LEN=512
    DROPOUT=0.1
    LEARNING_RATE=3e-4
    EPOCHS=int(input("epochs:"))

    device=torch.device('cuda:0')
    print(f'using device:{device}')
    model = MiniTransformer(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # 可选：学习率 warmup + 余弦退火，但简单起见先不用 scheduler

    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss, val_ppl = evaluate(model, valid_loader)
        print(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")
    while True:
        prompt=input("prompt(q is quit):")
        if prompt=='q':
            print('over')
            break
        else:
            output=generate(model,tokenizer,prompt,max_new_tokens=30)
            print(f'output:/n{output}')




