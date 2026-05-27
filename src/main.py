import torch
import torch.nn as nn
import torch.optim as optim

from data import oxfordIIIPet_loader,cifar100_loader
from models_ViT import VisionTransformer
from utils import set_seed, train, evaluate


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    set_seed(42)

    # Hyperparameters
    img_size = 224
    batch_size = 64
    num_workers = 16
    epochs = 100
    lr = 5e-4
    warmup_epochs = 5

    # Data
    train_loader, test_loader = cifar100_loader(
        size=img_size, batch_size=batch_size, num_workers=num_workers
    )

    # Model — ViT-Small scale
    num_classes = len(train_loader.dataset.classes)
    model = VisionTransformer(
        img_size=img_size, patch_size=16, in_channels=3,
        num_classes=num_classes, embed_dim=384, n_head=6,
        num_layers=6, d_ff=1024, dropout=0.1, device=device
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {total_params:,}')
    print(f'Number of classes: {num_classes}')

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)

    # Train (with warmup + cosine decay)
    train(model, train_loader, test_loader, criterion, optimizer, device,
          epochs, warmup_epochs=warmup_epochs)

    # Final evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f'Final test result - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%')


if __name__ == '__main__':
    main()
