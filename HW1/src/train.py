import os
import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from tqdm import tqdm
import wandb
from torchvision.transforms import v2
from dataset import get_dataloaders
from model import get_model


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train image classifier')
    
    # Data arguments
    parser.add_argument('--train_dir', type=str, default='../dataset/data/train',
                        help='Path to training data directory')
    parser.add_argument('--val_dir', type=str, default='../dataset/data/val',
                        help='Path to validation data directory')
    parser.add_argument('--num_classes', type=int, default=100,
                        help='Number of classes')
    
    # Model arguments
    parser.add_argument('--backbone', type=str, default='resnet50',
                        choices=['resnet50', 'resnet101', 'resnet152'],
                        help='Backbone architecture')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate before FC layer')
    parser.add_argument('--use_deeper_fc', action='store_true',
                        help='Use a deeper FC layer')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for FC layer')
    parser.add_argument('--lr_backbone', type=float, default=1e-4,
                        help='Learning rate for backbone')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'plateau', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs (freeze backbone)')
    parser.add_argument('--use_label_smoothing', action='store_true')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing factor (if --use_label_smoothing is set)')
    # Data augmentation arguments
    parser.add_argument('--use_RandAugment', action='store_true')
    parser.add_argument('--use_mixup_and_cutmix', action='store_true')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')
    
    # Misc arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Wandb arguments
    parser.add_argument('--wandb_project', type=str, default='vrdl-hw1',
                        help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Wandb run name')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging')
    
    return parser.parse_args()


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    epoch,
    use_mixup_and_cutmix=False,
    num_classes=100,
):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    cutmix = v2.CutMix(num_classes=num_classes)
    mixup = v2.MixUp(num_classes=num_classes)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        if use_mixup_and_cutmix:
            images, labels = cutmix_or_mixup(images, labels)
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        # MixUp/CutMix returns soft labels with shape [B, C].
        # For reporting training accuracy, map soft targets back to hard class indices.
        if labels.ndim == 2:
            target_for_acc = labels.argmax(dim=1)
        else:
            target_for_acc = labels
        correct += predicted.eq(target_for_acc).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss / (batch_idx + 1):.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, epoch):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss / (batch_idx + 1):.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, path):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_acc': best_acc
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, scheduler, path, device):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['best_acc']


def main():
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize wandb
    if not args.no_wandb:
        run_name = args.wandb_run_name or f"{args.backbone}_bs{args.batch_size}_lr{args.lr}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args)
        )
    
    # Create dataloaders
    print("\n" + "=" * 50)
    print("Loading datasets...")
    print("=" * 50)
    train_loader, val_loader = get_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
        use_RandAugment=args.use_RandAugment
    )
    
    # Create model
    print("\n" + "=" * 50)
    print("Creating model...")
    print("=" * 50)
    model = get_model(
        backbone=args.backbone,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        dropout=args.dropout,
        use_deeper_fc=args.use_deeper_fc
    )
    model = model.to(device)
    
    # Loss function
    if args.use_label_smoothing:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    param_groups = model.get_params_for_optimizer(
        lr_backbone=args.lr_backbone,
        lr_fc=args.lr
    )
    
    if args.optimizer == 'adam':
        optimizer = optim.Adam(param_groups, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(param_groups, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = None
    
    # Resume from checkpoint
    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        start_epoch, best_acc = load_checkpoint(model, optimizer, scheduler, args.resume, device)
        start_epoch += 1
        print(f"Resuming from epoch {start_epoch}, best acc: {best_acc:.2f}%")
    
    # Training loop
    print("\n" + "=" * 50)
    print("Starting training...")
    print("=" * 50)
    
    for epoch in range(start_epoch, args.epochs):
        # Warmup: freeze backbone for first few epochs
        if epoch < args.warmup_epochs:
            if epoch == 0:
                print(f"\nWarmup phase: Freezing backbone for {args.warmup_epochs} epochs")
                model.freeze_backbone()
        elif epoch == args.warmup_epochs:
            print(f"\nWarmup complete: Unfreezing backbone")
            model.unfreeze_backbone()
            # Reset parameter groups with proper learning rates
            param_groups = model.get_params_for_optimizer(
                lr_backbone=args.lr_backbone,
                lr_fc=args.lr
            )
            if args.optimizer == 'adam':
                optimizer = optim.Adam(param_groups, weight_decay=args.weight_decay)
            elif args.optimizer == 'adamw':
                optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
            elif args.optimizer == 'sgd':
                optimizer = optim.SGD(param_groups, momentum=0.9, weight_decay=args.weight_decay)

            if args.scheduler == 'cosine':
                scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
            elif args.scheduler == 'step':
                scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
            elif args.scheduler == 'plateau':
                scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
            else:
                scheduler = None
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            use_mixup_and_cutmix=args.use_mixup_and_cutmix,
            num_classes=args.num_classes,
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Update scheduler
        if scheduler and epoch >= args.warmup_epochs:
            if args.scheduler == 'plateau':
                scheduler.step(val_acc)
            else:
                scheduler.step()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        current_lr_fc = optimizer.param_groups[1]['lr'] if len(optimizer.param_groups) > 1 else current_lr
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  LR (backbone): {current_lr:.6f}, LR (FC): {current_lr_fc:.6f}")
        
        # Log to wandb
        if not args.no_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr_backbone': current_lr,
                'lr_fc': current_lr_fc
            })
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(args.save_dir, f'best_{args.backbone}.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, best_acc, best_path)
            print(f"New best accuracy: {best_acc:.2f}%")
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_freq == 0:
            ckpt_path = os.path.join(args.save_dir, f'{args.backbone}_epoch{epoch}.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, best_acc, ckpt_path)
    
    # Save final model
    final_path = os.path.join(args.save_dir, f'final_{args.backbone}.pth')
    save_checkpoint(model, optimizer, scheduler, args.epochs - 1, best_acc, final_path)
    
    print("\n" + "=" * 50)
    print("Training complete!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print("=" * 50)
    
    if not args.no_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
