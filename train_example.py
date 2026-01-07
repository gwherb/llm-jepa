"""
Example training script for GPT2-JEPA on the reversal curse task.

This script demonstrates how to:
1. Load the data using the custom dataloader
2. Initialize the GPT2-JEPA model
3. Run the training loop with automatic reverse generation
"""

import torch
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, GPT2Config
import os

from models.gpt2_jepa import GPT2WithJEPA, train_jepa, evaluate_jepa
from models.gpt2_jepa_config import JEPAConfig
from models.gpt2_jepa_dataloader import create_dataloaders, get_special_token_ids


def main():
    # ========== Configuration ==========
    # Paths
    data_dir = r'c:\Users\gwher\OneDrive\Desktop\llm-jepa\data\inversionidcomb10.50000.30000'
    train_path = os.path.join(data_dir, 'train.json')
    valid_path = os.path.join(data_dir, 'valid.json')
    vocab_path = os.path.join(data_dir, 'vocab.json')
    save_dir = r'c:\Users\gwher\OneDrive\Desktop\llm-jepa\checkpoints'

    # Model settings
    vocab_size = 16015  # Adjust based on your actual vocab size

    # JEPA hyperparameters
    jepa_config = JEPAConfig(
        lambda_jepa=1.0,       # JEPA loss weight
        gamma_ntp=1.0,         # NTP loss weight
        k_pred_tok=1,          # Number of predictor tokens
        loss_dropout=0.0,      # JEPA loss dropout (0.0 = no dropout)
        distance_metric='cosine',  # 'cosine', 'l2', or 'mse'
        use_jepa=True,         # Enable JEPA loss
    )

    # Training hyperparameters
    batch_size = 32
    num_epochs = 3
    learning_rate = 5e-5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*70)
    print("GPT2-JEPA Training for Reversal Curse")
    print("="*70)
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print()

    # ========== Load Special Tokens ==========
    print("Loading special tokens...")
    special_tokens = get_special_token_ids(vocab_path)
    pred_token_id = special_tokens['pred']
    mask_token_id = special_tokens['mask']
    first_relation_token_id = special_tokens['first_relation']

    print(f"  <PRED> token ID: {pred_token_id}")
    print(f"  <mask> token ID: {mask_token_id}")
    print(f"  <r_0> token ID: {first_relation_token_id}")
    print()

    # ========== Create Dataloaders ==========
    print("Creating dataloaders...")
    train_dataloader, valid_dataloader = create_dataloaders(
        train_path=train_path,
        valid_path=valid_path,
        batch_size=batch_size,
        num_workers=0,
        shuffle_train=True,
        valid_split='train',  # Use 'train' split from validation data
    )
    print(f"  Training batches: {len(train_dataloader)}")
    print(f"  Validation batches: {len(valid_dataloader)}")
    print()

    # ========== Initialize Model ==========
    print("Initializing GPT2 model...")

    # Option 1: Load pretrained GPT2
    # base_model = GPT2LMHeadModel.from_pretrained('gpt2')
    # base_model.resize_token_embeddings(vocab_size)

    # Option 2: Create from scratch with custom config
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
    )
    base_model = GPT2LMHeadModel(config)

    # Wrap with JEPA
    model = GPT2WithJEPA(base_model=base_model, jepa_config=jepa_config)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print()

    # ========== Initialize Optimizer ==========
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Optional: Learning rate scheduler
    # from torch.optim.lr_scheduler import CosineAnnealingLR
    # scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs*len(train_dataloader))
    scheduler = None

    # ========== Training ==========
    print("Starting training...")
    os.makedirs(save_dir, exist_ok=True)

    training_stats = train_jepa(
        model=model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device,
        pred_token_id=pred_token_id,
        first_relation_token_id=first_relation_token_id,
        eval_dataloader=valid_dataloader,
        scheduler=scheduler,
        log_interval=100,
        save_path=save_dir,
    )

    # ========== Save Final Model ==========
    final_model_path = os.path.join(save_dir, 'final_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'jepa_config': jepa_config,
        'training_stats': training_stats,
    }, final_model_path)
    print(f"\nFinal model saved to {final_model_path}")

    # ========== Final Evaluation ==========
    print("\nRunning final evaluation on test set...")
    test_dataloader = create_dataloaders(
        train_path=train_path,  # Dummy, not used
        valid_path=valid_path,
        batch_size=batch_size,
        valid_split='test',  # Use test split
    )[1]

    test_metrics = evaluate_jepa(
        model=model,
        eval_dataloader=test_dataloader,
        device=device,
        pred_token_id=pred_token_id,
        first_relation_token_id=first_relation_token_id,
    )

    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test NTP Loss: {test_metrics['ntp_loss']:.4f}")
    print(f"Test JEPA Loss: {test_metrics['jepa_loss']:.4f}")

    print("\nTraining completed successfully!")


if __name__ == '__main__':
    main()
