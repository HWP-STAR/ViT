import time
import random
import torch
import numpy as np


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
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
            print(f'Epoch {epoch} | Batch {batch_idx}/{len(loader)} | Loss: {cur_loss:.4f} | Acc: {cur_acc:.2f}% | Time: {elapsed:.2f}s')

    avg_loss = total_loss / len(loader)
    acc = 100. * correct / total
    return avg_loss, acc


def train(model, train_loader, test_loader, criterion, optimizer, device, epochs):
    best_acc = 0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        print(f'====> Epoch {epoch} Train | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%')

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
