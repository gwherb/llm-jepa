"""
Main training script for GPT2-JEPA on Ohio Supercomputer

This script is designed to run on OSC with SLURM job scheduling.
It supports:
- Multi-GPU training (though currently uses 1 GPU)
- Checkpoint saving and resuming
- Comprehensive logging
- Evaluation during training
"""

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ConstantLR
from transformers import GPT2LMHeadModel, GPT2Config
import os
import argparse
import json
from datetime import datetime

from models.gpt2_jepa import GPT2WithJEPA, train_jepa, evaluate_jepa
from models.gpt2_jepa_config import JEPAConfig
from models.gpt2_jepa_dataloader import create_dataloaders, get_special_token_ids


def parse_args():
    parser = argparse.ArgumentParser(description='Train GPT2 with JEPA')

    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing train.json, valid.json, and vocab.json')

    # Model arguments
    parser.add_argument('--vocab_size', type=int, default=16015,
                        help='Vocabulary size')
    parser.add_argument('--n_positions', type=int, default=128,
                        help='Maximum sequence length')
    parser.add_argument('--n_embd', type=int, default=768,
                        help='Embedding dimension')
    parser.add_argument('--n_layer', type=int, default=12,
                        help='Number of transformer layers')
    parser.add_argument('--n_head', type=int, default=12,
                        help='Number of attention heads')

    # JEPA arguments
    parser.add_argument('--lambda_jepa', type=float, default=1.0,
                        help='JEPA loss weight')
    parser.add_argument('--gamma_ntp', type=float, default=1.0,
                        help='NTP loss weight')
    parser.add_argument('--k_pred_tok', type=int, default=1,
                        help='Number of predictor tokens')
    parser.add_argument('--loss_dropout', type=float, default=0.0,
                        help='JEPA loss dropout (0.0 to 1.0)')
    parser.add_argument('--distance_metric', type=str, default='cosine',
                        choices=['cosine', 'l2', 'mse'],
                        help='Distance metric for JEPA loss')
    parser.add_argument('--use_jepa', action='store_true', default=True,
                        help='Enable JEPA loss')

    # Training arguments
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size per GPU')
    parser.add_argument('--eval_batch_size', type=int, default=32,
                        help='Evaluation batch size')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Max gradient norm for clipping')
    parser.add_argument('--warmup_steps', type=int, default=0,
                        help='Number of warmup steps')

    # Checkpoint arguments
    parser.add_argument('--save_steps', type=int, default=10000,
                        help='Save checkpoint every N steps')
    parser.add_argument('--eval_steps', type=int, default=5000,
                        help='Evaluate every N steps')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Log every N steps')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--valid_split', type=str, default='train',
                        choices=['train', 'atomic', 'test'],
                        help='Validation split to use')
    parser.add_argument('--fp16', action='store_true',
                        help='Use mixed precision training')

    return parser.parse_args()


def setup_logging(output_dir):
    """Create output directory and setup logging"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)

    # Create log file
    log_file = os.path.join(output_dir, 'training.log')
    return log_file


def save_checkpoint(model, optimizer, scheduler, epoch, step, args, output_dir, is_best=False):
    """Save model checkpoint"""
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    checkpoint_name = f'checkpoint_epoch{epoch}_step{step}.pt'
    if is_best:
        checkpoint_name = 'best_model.pt'

    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'jepa_config': model.jepa_config,
        'args': vars(args),
    }, checkpoint_path)

    print(f"Checkpoint saved to {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load model checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('step', 0)

    print(f"Resumed from epoch {epoch}, step {step}")
    return epoch, step


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*70)
    print("GPT2-JEPA Training - Ohio Supercomputer Center")
    print("="*70)
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Training started at: {datetime.now()}")
    print("="*70)
    print()

    # Setup logging
    log_file = setup_logging(args.output_dir)

    # Save arguments
    args_file = os.path.join(args.output_dir, 'training_args.json')
    with open(args_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Training arguments saved to {args_file}")

    # Print configuration
    print("Configuration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Model: {args.n_layer} layers, {args.n_embd} dim, {args.n_head} heads")
    print(f"  JEPA: lambda={args.lambda_jepa}, gamma={args.gamma_ntp}, k={args.k_pred_tok}")
    print(f"  JEPA metric: {args.distance_metric}, dropout={args.loss_dropout}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Epochs: {args.num_epochs}")
    print()

    # Load special tokens
    vocab_path = os.path.join(args.data_dir, 'vocab.json')
    print("Loading special tokens...")
    special_tokens = get_special_token_ids(vocab_path)
    pred_token_id = special_tokens['pred']
    first_relation_token_id = special_tokens['first_relation']
    print(f"  <PRED> token: {pred_token_id}")
    print(f"  <r_0> token: {first_relation_token_id}")
    print()

    # Create dataloaders
    train_path = os.path.join(args.data_dir, 'train.json')
    valid_path = os.path.join(args.data_dir, 'valid.json')

    print("Creating dataloaders...")
    train_dataloader, valid_dataloader = create_dataloaders(
        train_path=train_path,
        valid_path=valid_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle_train=True,
        valid_split=args.valid_split,
    )
    print(f"  Training batches per epoch: {len(train_dataloader)}")
    print(f"  Validation batches: {len(valid_dataloader)}")
    print()

    # Create model
    print("Initializing model...")
    jepa_config = JEPAConfig(
        lambda_jepa=args.lambda_jepa,
        gamma_ntp=args.gamma_ntp,
        k_pred_tok=args.k_pred_tok,
        loss_dropout=args.loss_dropout,
        distance_metric=args.distance_metric,
        use_jepa=args.use_jepa,
    )

    gpt2_config = GPT2Config(
        vocab_size=args.vocab_size,
        n_positions=args.n_positions,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        loss_type='ForCausalLMLoss',  # Suppress warning
    )

    base_model = GPT2LMHeadModel(gpt2_config)
    model = GPT2WithJEPA(base_model=base_model, jepa_config=jepa_config)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print()

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Create scheduler
    scheduler = None
    if args.warmup_steps > 0:
        total_steps = len(train_dataloader) * args.num_epochs
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - args.warmup_steps)

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume_from:
        start_epoch, _ = load_checkpoint(args.resume_from, model, optimizer, scheduler)

    # Mixed precision training
    scaler = None
    if args.fp16 and torch.cuda.is_available():
        scaler = torch.amp.GradScaler('cuda')
        print("Using mixed precision training (FP16)")
        print()

    # Training
    print("Starting training...")
    print("="*70)

    training_stats = train_jepa(
        model=model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        num_epochs=args.num_epochs,
        device=device,
        pred_token_id=pred_token_id,
        first_relation_token_id=first_relation_token_id,
        eval_dataloader=valid_dataloader,
        scheduler=scheduler,
        log_interval=args.log_interval,
        save_path=args.output_dir,
    )

    # Save final model
    final_checkpoint = save_checkpoint(
        model, optimizer, scheduler,
        args.num_epochs, len(train_dataloader) * args.num_epochs,
        args, args.output_dir, is_best=True
    )

    print()
    print("="*70)
    print("Training completed!")
    print(f"Final model saved to {final_checkpoint}")
    print(f"Training finished at: {datetime.now()}")
    print("="*70)


if __name__ == '__main__':
    main()
