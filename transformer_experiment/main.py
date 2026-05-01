# main.py
import torch
from try_model import MiniTransformer   # 假设原模型定义在该文件中
from data import (
    load_wikitext_data, WordTokenizer,
    create_dataloaders
)
from train import train_epoch, evaluate, generate
from utils import set_seed

def main():
    # 超参数
    EMBED_DIM = 512
    NUM_HEADS = 4
    NUM_LAYERS = 16
    MAX_SEQ_LEN = 512
    DROPOUT = 0.1
    LEARNING_RATE = 3e-4
    BATCH_SIZE = 64
    VOCAB_SIZE_LIMIT = 5000
    MAX_LEN = 128          # 用于数据集的截断长度
    EPOCHS = int(input("epochs: "))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    set_seed(42)

    # 1. 加载数据
    train_texts, valid_texts, _ = load_wikitext_data(cache_dir="../../data")

    # 2. 构建词表
    tokenizer = WordTokenizer(vocab_size=VOCAB_SIZE_LIMIT)
    tokenizer.build_vocab(train_texts)
    vocab_size = tokenizer.vocab_size
    print(f"Final vocab size: {vocab_size}")

    # 3. 创建 DataLoader
    train_loader, valid_loader = create_dataloaders(
        train_texts, valid_texts, tokenizer,
        batch_size=BATCH_SIZE, max_len=MAX_LEN
    )
    print("DataLoader ready.")

    # 4. 初始化模型
    model = MiniTransformer(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT
    ).to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 5. 训练循环
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, device, vocab_size)
        val_loss, val_ppl = evaluate(model, valid_loader, device, vocab_size)
        print(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")

    # 6. 交互生成
    while True:
        prompt = input("prompt (q to quit): ")
        if prompt == 'q':
            print('over')
            break
        output = generate(
            model, tokenizer, prompt, device,
            max_seq_len=MAX_SEQ_LEN, max_new_tokens=30, temperature=1.0
        )
        print(f'output:\n{output}')

if __name__ == "__main__":
    main()
