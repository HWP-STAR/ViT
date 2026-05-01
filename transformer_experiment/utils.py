# train_eval.py
import numpy as np
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional, Callable, Union

def train_epoch(
    model: Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    device: torch.device,
    criterion: Callable = F.cross_entropy,
    scheduler: Optional[_LRScheduler] = None,
    clip_grad_norm: float = 1.0,
    log_interval: int = 100,
    ignore_index: int = -100,
) -> float:
    """
    训练一个epoch。

    Args:
        model: 待训练的模型
        dataloader: 训练数据加载器，每个batch包含 (inputs, labels)
        optimizer: 优化器
        device: 设备 (cpu/cuda)
        criterion: 损失函数，默认为 cross_entropy，原型为 func(logits, targets, ignore_index)
        scheduler: 学习率调度器，在每个batch后步进（若提供）
        clip_grad_norm: 梯度裁剪的范数阈值，设为 None 或 0 禁用
        log_interval: 每隔多少batch打印一次损失
        ignore_index: 损失计算时忽略的标签值

    Returns:
        平均损失 (float)
    """
    model.train()
    total_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        if inputs.size(0) == 0:
            continue

        optimizer.zero_grad()
        logits = model(inputs)  # shape: (B, T, vocab_size)

        # 展平并计算损失
        loss = criterion(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=ignore_index,
        )
        loss.backward()

        if clip_grad_norm and clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

        if log_interval and batch_idx % log_interval == 0:
            print(f"  Batch {batch_idx}, loss: {loss.item():.4f}")

    return total_loss / len(dataloader)


def evaluate(
    model: Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: Callable = F.cross_entropy,
    ignore_index: int = -100,
) -> tuple[float, float]:
    """
    评估模型，返回平均损失和困惑度。

    Args:
        model: 待评估的模型
        dataloader: 验证/测试数据加载器
        device: 设备
        criterion: 损失函数，注意这里会使用 reduction='sum' 来累加
        ignore_index: 忽略的标签值

    Returns:
        (avg_loss, perplexity): 平均每个token的损失，以及困惑度（exp(avg_loss)）
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            if inputs.size(0) == 0:
                continue

            logits = model(inputs)
            # 手动求和，避免除以batch size
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=ignore_index,
                reduction='sum',
            )
            total_loss += loss.item()
            total_tokens += (labels != ignore_index).sum().item()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    # 防止溢出，限制 avg_loss 最大值
    if avg_loss < 100:
        perplexity = math.exp(avg_loss)
    else:
        perplexity = float('inf')
    return avg_loss, perplexity

# utils.py
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
