# train.py
import torch
import torch.nn.functional as F
import math

def train_epoch(model, dataloader, optimizer, device, vocab_size, scheduler=None, clip_grad=1.0):
    model.train()
    total_loss = 0
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        if inputs.size(0) == 0:
            continue
        optimizer.zero_grad()
        logits = model(inputs)                     # (B, T, vocab_size)
        loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        if scheduler:
            scheduler.step()
        total_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}, loss: {loss.item():.4f}")
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, vocab_size):
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            if inputs.size(0) == 0:
                continue
            logits = model(inputs)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size), labels.view(-1),
                ignore_index=-100, reduction='sum'
            )
            total_loss += loss.item()
            total_tokens += (labels != -100).sum().item()
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    return avg_loss, perplexity


def generate(model, tokenizer, prompt, device, max_seq_len, max_new_tokens=50, temperature=1.0):
    model.eval()
    input_ids = tokenizer.encode(prompt, max_len=max_seq_len)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
    pad_id = tokenizer.word2idx[tokenizer.pad_token]

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids)                     # (1, T, vocab_size)
        next_token_logits = logits[0, -1, :] / temperature
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        if next_token == pad_id:
            break
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=1)
        if input_ids.size(1) >= max_seq_len:
            break
    generated = tokenizer.decode(input_ids[0].tolist())
    return generated
