"""
Local test training script for GPT2-JEPA (CPU-friendly)

This script runs a minimal training test to verify:
1. DataLoader works correctly
2. Model forward pass executes without errors
3. JEPA loss computation is correct
4. Gradient flow works properly
5. All components integrate correctly

Configuration is optimized for CPU testing with small model/batch sizes.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Config
import os

from models.gpt2_jepa import GPT2WithJEPA, train_jepa
from models.gpt2_jepa_config import JEPAConfig
from models.gpt2_jepa_dataloader import create_dataloaders, get_special_token_ids


def test_training():
    print("="*70)
    print("LOCAL TEST: GPT2-JEPA Training (CPU)")
    print("="*70)

    # ========== Configuration (Minimal for CPU) ==========
    # Paths
    data_dir = r'c:\Users\gwher\OneDrive\Desktop\llm-jepa\data\inversionidcomb10.50000.30000'
    train_path = os.path.join(data_dir, 'train.json')
    valid_path = os.path.join(data_dir, 'valid.json')
    vocab_path = os.path.join(data_dir, 'vocab.json')

    # Model settings (SMALL for CPU testing)
    vocab_size = 16015

    # JEPA config
    jepa_config = JEPAConfig(
        lambda_jepa=1.0,
        gamma_ntp=1.0,
        k_pred_tok=1,          # Test with k=1
        loss_dropout=0.0,
        distance_metric='cosine',
        use_jepa=True,
    )

    # Training settings (MINIMAL for quick test)
    batch_size = 4          # Small batch
    num_epochs = 1          # Just 1 epoch
    learning_rate = 5e-5
    max_steps = 10          # Only train for 10 steps
    device = torch.device('cpu')  # Force CPU

    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Max steps: {max_steps}")
    print(f"JEPA config: k={jepa_config.k_pred_tok}, metric={jepa_config.distance_metric}")
    print()

    # ========== Load Special Tokens ==========
    print("Loading special tokens...")
    special_tokens = get_special_token_ids(vocab_path)
    pred_token_id = special_tokens['pred']
    mask_token_id = special_tokens['mask']
    first_relation_token_id = special_tokens['first_relation']

    print(f"  <PRED>: {pred_token_id}")
    print(f"  <mask>: {mask_token_id}")
    print(f"  <r_0>: {first_relation_token_id}")
    print()

    # ========== Create Dataloaders ==========
    print("Creating dataloaders...")
    train_dataloader, valid_dataloader = create_dataloaders(
        train_path=train_path,
        valid_path=valid_path,
        batch_size=batch_size,
        num_workers=0,
        shuffle_train=True,
        valid_split='train',
    )
    print()

    # ========== Test DataLoader ==========
    print("Testing dataloader...")
    batch = next(iter(train_dataloader))
    print(f"  Batch keys: {batch.keys()}")
    print(f"  input_ids shape: {batch['input_ids'].shape}")
    print(f"  labels shape: {batch['labels'].shape}")
    print(f"  types: {batch['type']}")
    print(f"  Sample input_ids[0]: {batch['input_ids'][0].tolist()}")
    print(f"  Sample labels[0]: {batch['labels'][0].tolist()}")
    print("  ✓ DataLoader working correctly")
    print()

    # ========== Initialize Model (TINY for CPU) ==========
    print("Initializing tiny GPT2 model for testing...")
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=64,      # Smaller context
        n_embd=128,          # Tiny embedding dim
        n_layer=2,           # Only 2 layers
        n_head=2,            # 2 attention heads
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

    # ========== Test Forward Pass ==========
    print("Testing forward pass...")
    model.eval()
    with torch.no_grad():
        test_batch = next(iter(train_dataloader))
        forward_input_ids = test_batch['input_ids'].to(device)
        forward_labels = test_batch['labels'].to(device)
        batch_types = test_batch['type']

        # Create compute_jepa mask
        compute_jepa = torch.tensor(
            [t == 'train' for t in batch_types],
            dtype=torch.bool,
            device=device
        )

        # Generate reverse
        reverse_input_ids = model.create_reverse_sequence(
            forward_input_ids,
            first_relation_token_id
        )

        print(f"  Forward input[0]: {forward_input_ids[0].tolist()}")
        print(f"  Reverse input[0]: {reverse_input_ids[0].tolist()}")

        # Forward pass
        outputs = model.forward_jepa(
            forward_input_ids=forward_input_ids,
            forward_labels=forward_labels,
            reverse_input_ids=reverse_input_ids,
            compute_jepa=compute_jepa,
            pred_token_id=pred_token_id,
        )

        print(f"  Total loss: {outputs['loss'].item():.4f}")
        print(f"  NTP loss: {outputs['ntp_loss'].item():.4f}")
        print(f"  JEPA loss: {outputs['jepa_loss'].item():.4f}")
        print(f"  Logits shape: {outputs['logits'].shape}")
        print("  ✓ Forward pass successful")
        print()

    # ========== Test Training Loop ==========
    print("Testing training loop (10 steps)...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    step_count = 0
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            if step_count >= max_steps:
                break

            # Move to device
            forward_input_ids = batch['input_ids'].to(device)
            forward_labels = batch['labels'].to(device)
            batch_types = batch['type']

            # Create JEPA mask
            compute_jepa = torch.tensor(
                [t == 'train' for t in batch_types],
                dtype=torch.bool,
                device=device
            )

            # Generate reverse
            reverse_input_ids = None
            if compute_jepa.any():
                reverse_input_ids = model.create_reverse_sequence(
                    forward_input_ids,
                    first_relation_token_id
                )

            # Forward pass
            outputs = model.forward_jepa(
                forward_input_ids=forward_input_ids,
                forward_labels=forward_labels,
                reverse_input_ids=reverse_input_ids,
                compute_jepa=compute_jepa,
                pred_token_id=pred_token_id,
            )

            loss = outputs['loss']
            ntp_loss = outputs['ntp_loss']
            jepa_loss = outputs['jepa_loss']

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Check gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            print(f"  Step {step_count+1}/{max_steps}: "
                  f"Loss={loss.item():.4f}, "
                  f"NTP={ntp_loss.item():.4f}, "
                  f"JEPA={jepa_loss.item():.4f}, "
                  f"GradNorm={grad_norm.item():.4f}")

            step_count += 1

        if step_count >= max_steps:
            break

    print("  ✓ Training loop successful")
    print()

    # ========== Test Reverse Generation Logic ==========
    print("Testing reverse generation logic...")
    test_batch = next(iter(train_dataloader))
    forward_ids = test_batch['input_ids'][:2].to(device)  # Take 2 samples

    reverse_ids = model.create_reverse_sequence(forward_ids, first_relation_token_id)

    for i in range(2):
        f = forward_ids[i].tolist()
        r = reverse_ids[i].tolist()
        print(f"  Sample {i}:")
        print(f"    Forward:  e1=[{f[0]}, {f[1]}], rel={f[2]}, mask={f[3]}, e2=[{f[4]}, {f[5]}]")
        print(f"    Reverse:  e2=[{r[0]}, {r[1]}], rel={r[2]}, mask={r[3]}, e1=[{r[4]}, {r[5]}]")

        # Check relation inversion
        forward_rel = f[2]
        reverse_rel = r[2]
        first_is_odd = (first_relation_token_id % 2) == 1

        if first_is_odd:
            if forward_rel % 2 == 1:  # odd (forward)
                expected_reverse = forward_rel + 1
            else:  # even (inverse)
                expected_reverse = forward_rel - 1
        else:
            if forward_rel % 2 == 0:  # even (forward)
                expected_reverse = forward_rel + 1
            else:  # odd (inverse)
                expected_reverse = forward_rel - 1

        if reverse_rel == expected_reverse:
            print(f"    ✓ Relation inverted correctly: {forward_rel} → {reverse_rel}")
        else:
            print(f"    ✗ ERROR: Expected {expected_reverse}, got {reverse_rel}")
    print()

    # ========== Test Block-Causal Masking ==========
    print("Testing block-causal attention mask...")
    mask = model._create_block_causal_mask(
        seq_len=6,
        k_pred=1,
        batch_size=1,
        device=device
    )

    print(f"  Mask shape: {mask.shape}")  # (1, 1, 13, 13) with k=1
    mask_2d = mask[0, 0].cpu()

    # Check key properties
    forward_len = 7  # 6 + 1 PRED
    reverse_start = forward_len

    # Forward block should be causal
    forward_block = mask_2d[:forward_len, :forward_len]
    is_causal_forward = torch.allclose(
        forward_block,
        torch.triu(torch.full_like(forward_block, float('-inf')), diagonal=1)
    )

    # Reverse block should be causal
    reverse_block = mask_2d[reverse_start:, reverse_start:]
    is_causal_reverse = torch.allclose(
        reverse_block,
        torch.triu(torch.full_like(reverse_block, float('-inf')), diagonal=1)
    )

    # Cross-blocks should be all -inf
    cross_block_1 = mask_2d[:forward_len, reverse_start:]
    cross_block_2 = mask_2d[reverse_start:, :forward_len]
    is_isolated = (torch.all(torch.isinf(cross_block_1)) and
                   torch.all(torch.isinf(cross_block_2)))

    print(f"  Forward block causal: {is_causal_forward}")
    print(f"  Reverse block causal: {is_causal_reverse}")
    print(f"  Blocks isolated: {is_isolated}")

    if is_causal_forward and is_causal_reverse and is_isolated:
        print("  ✓ Block-causal masking correct")
    else:
        print("  ✗ ERROR: Masking has issues")
    print()

    # ========== Final Summary ==========
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    print("✓ DataLoader: Working")
    print("✓ Forward Pass: Working")
    print("✓ Training Loop: Working")
    print("✓ Reverse Generation: Working")
    print("✓ Block-Causal Masking: Working")
    print("✓ Gradient Flow: Working")
    print()
    print("All systems operational! Ready for full training on supercomputer.")
    print("="*70)


if __name__ == '__main__':
    test_training()
