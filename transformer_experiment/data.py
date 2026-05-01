# data.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_from_disk
from collections import Counter

class WordTokenizer:
    def __init__(self, vocab_size=5000, unk_token="<UNK>", pad_token="<PAD>"):
        self.vocab_size_limit = vocab_size
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.word2idx = {}
        self.idx2word = {}

    def build_vocab(self, texts):
        word_counts = Counter()
        for text in texts:
            words = text.strip().split()
            word_counts.update(words)

        most_common = word_counts.most_common(self.vocab_size_limit - 2)
        self.word2idx[self.pad_token] = 0
        self.word2idx[self.unk_token] = 1
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        print(f"Vocabulary built: {len(self.word2idx)} words (including PAD/UNK)")

    def encode(self, text, max_len=None):
        words = text.strip().split()
        indices = [self.word2idx.get(word, self.word2idx[self.unk_token]) for word in words]
        if max_len is not None:
            if len(indices) > max_len:
                indices = indices[:max_len]
            else:
                indices += [self.word2idx[self.pad_token]] * (max_len - len(indices))
        return indices

    def decode(self, indices):
        words = [self.idx2word.get(idx, self.unk_token) for idx in indices]
        return " ".join(words)

    @property
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
        ids = self.tokenizer.encode(text, max_len=self.max_len)
        if len(ids) > 1:
            input_ids = torch.tensor(ids[:-1], dtype=torch.long)
            labels = torch.tensor(ids[1:], dtype=torch.long)
        else:
            input_ids = torch.tensor([], dtype=torch.long)
            labels = torch.tensor([], dtype=torch.long)
        return input_ids, labels


def create_collate_fn(tokenizer):
    """返回针对该 tokenizer 的 collate 函数"""
    pad_idx = tokenizer.word2idx[tokenizer.pad_token]

    def collate_fn(batch):
        max_len = max([inp.size(0) for inp, _ in batch if inp.size(0) > 0])
        if max_len == 0:
            return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)

        padded_inputs = []
        padded_labels = []
        for inp, lbl in batch:
            if inp.size(0) == 0:
                continue
            pad_len = max_len - inp.size(0)
            inp_padded = F.pad(inp, (0, pad_len), value=pad_idx)
            lbl_padded = F.pad(lbl, (0, pad_len), value=-100)   # -100 被交叉熵忽略
            padded_inputs.append(inp_padded)
            padded_labels.append(lbl_padded)
        return torch.stack(padded_inputs), torch.stack(padded_labels)
    return collate_fn


def load_wikitext_data(cache_dir="../../data"):
    """返回 (train_texts, valid_texts, test_texts)"""
    dataset = load_from_disk("../../data")

    train_texts = [ex['text'] for ex in dataset['train'] if ex['text'].strip()]
    valid_texts = [ex['text'] for ex in dataset['validation'] if ex['text'].strip()]
    test_texts  = [ex['text'] for ex in dataset['test'] if ex['text'].strip()]
    print(f"Train samples: {len(train_texts)}, Valid: {len(valid_texts)}, Test: {len(test_texts)}")
    return train_texts, valid_texts, test_texts


def create_dataloaders(train_texts, valid_texts, tokenizer, batch_size=64, max_len=128):
    """返回 train_loader, valid_loader"""
    train_dataset = TextDataset(train_texts, tokenizer, max_len=max_len)
    valid_dataset = TextDataset(valid_texts, tokenizer, max_len=max_len)

    collate_fn = create_collate_fn(tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    return train_loader, valid_loader
