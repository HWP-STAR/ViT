import time
import random
import math
import torch
import numpy as np
from torch.optim.lr_scheduler import LRScheduler


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class WarmupCosineLR(LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch + 1
        if epoch <= self.warmup_epochs:
            alpha = epoch / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]
        progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [base_lr * factor for base_lr in self.base_lrs]


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, warmup_steps=None, total_steps=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    start_time = time.time()

    for batch_idx, (data, target) in enumerate(loader, 1):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % 100 == 0:
            cur_loss = total_loss / batch_idx
            cur_acc = 100. * correct / total
            elapsed = time.time() - start_time
            lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch} | Batch {batch_idx}/{len(loader)} | Loss: {cur_loss:.4f} | Acc: {cur_acc:.2f}% | LR: {lr:.2e} | Time: {elapsed:.2f}s')

    avg_loss = total_loss / len(loader)
    acc = 100. * correct / total
    return avg_loss, acc


def train(model, train_loader, test_loader, criterion, optimizer, device, epochs, warmup_epochs=5):
    best_acc = 0
    scheduler = WarmupCosineLR(optimizer, warmup_epochs, epochs)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f'====> Epoch {epoch} Train | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | LR: {current_lr:.2e}')

        if test_loader is not None:
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            print(f'====> Epoch {epoch} Test  | Loss: {test_loss:.4f} | Acc: {test_acc:.2f}%')
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), 'best_model.pth')
                print(f'====> Saved best model with Acc: {best_acc:.2f}%')

    print(f'Training complete. Best test accuracy: {best_acc:.2f}%')
    return best_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)

        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    avg_loss = total_loss / len(loader)
    acc = 100. * correct / total
    return avg_loss, acc


def save_checkpoint(model, optimizer, epoch, path='checkpoint.pth'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f'Checkpoint saved to {path}')


def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f'Checkpoint loaded from {path} (epoch {checkpoint["epoch"]})')
    return checkpoint['epoch']
