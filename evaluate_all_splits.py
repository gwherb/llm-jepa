"""
Comprehensive evaluation script for GPT2-JEPA

Evaluates the model on all three validation splits:
- 'train': Subsample of training data (checks overfitting)
- 'atomic': Samples with one direction seen (partial generalization)
- 'test': Held-out reverse directions (TRUE reversal curse test)
"""

import torch
from transformers import GPT2LMHeadModel
import argparse
import json

from models.gpt2_jepa import GPT2WithJEPA, evaluate_jepa
from models.gpt2_jepa_config import JEPAConfig
from models.gpt2_jepa_dataloader import create_dataloaders, get_special_token_ids


def evaluate_all_splits(checkpoint_path, data_dir, batch_size=32, device=None, per_layer=False):
    """
    Evaluate model on all three validation splits.

    Args:
        checkpoint_path: Path to model checkpoint
        data_dir: Directory containing data files
        batch_size: Batch size for evaluation
        device: Device to use (defaults to cuda if available)
        per_layer: Whether to compute per-layer MRR analysis (expensive)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*70)
    print("GPT2-JEPA Comprehensive Evaluation")
    print("="*70)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Data directory: {data_dir}")
    print(f"Device: {device}")
    print()

    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Get config from checkpoint
    jepa_config = checkpoint.get('jepa_config')
    args = checkpoint.get('args', {})

    print(f"JEPA config: lambda={jepa_config.lambda_jepa}, gamma={jepa_config.gamma_ntp}, k={jepa_config.k_pred_tok}")
    print()

    # Load special tokens
    vocab_path = f"{data_dir}/vocab.json"
    special_tokens = get_special_token_ids(vocab_path)
    pred_token_id = special_tokens['pred']
    first_relation_token_id = special_tokens['first_relation']

    print(f"Special tokens: PRED={pred_token_id}, r_0={first_relation_token_id}")
    print()

    # Initialize model
    print("Initializing model...")
    from transformers import GPT2Config

    gpt2_config = GPT2Config(
        vocab_size=args.get('vocab_size', 16015),
        n_positions=args.get('n_positions', 128),
        n_embd=args.get('n_embd', 768),
        n_layer=args.get('n_layer', 12),
        n_head=args.get('n_head', 12),
    )

    base_model = GPT2LMHeadModel(gpt2_config)
    model = GPT2WithJEPA(base_model=base_model, jepa_config=jepa_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    print()

    # Evaluate on all three splits
    results = {}

    for split_name in ['train', 'atomic', 'test']:
        print("="*70)
        print(f"Evaluating on '{split_name}' split")
        print("="*70)

        # Create dataloader for this split
        train_path = f"{data_dir}/train.json"
        valid_path = f"{data_dir}/valid.json"

        _, eval_dataloader = create_dataloaders(
            train_path=train_path,
            valid_path=valid_path,
            batch_size=batch_size,
            num_workers=0,
            shuffle_train=False,
            valid_split=split_name,
        )

        print(f"Evaluating {len(eval_dataloader)} batches...")
        if per_layer:
            print("(Computing per-layer MRR - this may take longer)")

        # Evaluate
        metrics = evaluate_jepa(
            model=model,
            eval_dataloader=eval_dataloader,
            device=device,
            pred_token_id=pred_token_id,
            first_relation_token_id=first_relation_token_id,
            compute_mrr=True,
            compute_per_layer=per_layer,
        )

        results[split_name] = metrics

        print(f"\nResults for '{split_name}' split:")
        print(f"  Total Loss:      {metrics['loss']:.4f}")
        print(f"  NTP Loss:        {metrics['ntp_loss']:.4f}")
        print(f"  JEPA Loss:       {metrics['jepa_loss']:.4f}")
        print(f"  MRR (avg):       {metrics['mrr']:.4f}")
        print(f"  MRR (token 1):   {metrics['mrr_token1']:.4f}")
        print(f"  MRR (token 2):   {metrics['mrr_token2']:.4f}")
        print(f"  Joint Log-Lik:   {metrics['joint_log_likelihood']:.4f}")
        print(f"  Joint Accuracy:  {metrics['joint_accuracy']:.4f}")
        print()

    # Summary comparison
    print("="*115)
    print("SUMMARY")
    print("="*115)
    print()
    print("Split       | NTP Loss  | MRR (avg) | MRR (tok1) | MRR (tok2) | Joint LL | Joint Acc | Notes")
    print("-" * 115)
    print(f"train       | {results['train']['ntp_loss']:9.4f} | {results['train']['mrr']:9.4f} | {results['train']['mrr_token1']:10.4f} | {results['train']['mrr_token2']:10.4f} | {results['train']['joint_log_likelihood']:8.4f} | {results['train']['joint_accuracy']:9.4f} | (overfitting)")
    print(f"atomic      | {results['atomic']['ntp_loss']:9.4f} | {results['atomic']['mrr']:9.4f} | {results['atomic']['mrr_token1']:10.4f} | {results['atomic']['mrr_token2']:10.4f} | {results['atomic']['joint_log_likelihood']:8.4f} | {results['atomic']['joint_accuracy']:9.4f} | (one dir)")
    print(f"test        | {results['test']['ntp_loss']:9.4f} | {results['test']['mrr']:9.4f} | {results['test']['mrr_token1']:10.4f} | {results['test']['mrr_token2']:10.4f} | {results['test']['joint_log_likelihood']:8.4f} | {results['test']['joint_accuracy']:9.4f} | (REVERSAL)")
    print()

    # Analysis
    print("ANALYSIS:")
    print("-" * 70)

    # Check overfitting
    train_vs_test_gap = results['test']['loss'] - results['train']['loss']
    print(f"1. Overfitting check:")
    print(f"   Test loss - Train loss = {train_vs_test_gap:.4f}")
    if train_vs_test_gap < 0.5:
        print(f"   ✓ Good generalization (gap < 0.5)")
    elif train_vs_test_gap < 1.0:
        print(f"   ⚠ Moderate gap (0.5 < gap < 1.0)")
    else:
        print(f"   ✗ Possible overfitting (gap > 1.0)")
    print()

    # Check JEPA effectiveness
    print(f"2. JEPA embedding alignment:")
    print(f"   Train JEPA loss: {results['train']['jepa_loss']:.6f}")
    if results['train']['jepa_loss'] < 0.01:
        print(f"   ✓ Excellent alignment (< 0.01)")
    elif results['train']['jepa_loss'] < 0.05:
        print(f"   ✓ Good alignment (< 0.05)")
    else:
        print(f"   ⚠ Moderate alignment (> 0.05)")
    print()

    # Reversal curse performance
    print(f"3. Reversal curse test (KEY METRIC):")
    print(f"   Test set MRR: {results['test']['mrr']:.4f}")
    print(f"   Test set loss: {results['test']['loss']:.4f}")
    print(f"   This measures performance on NEVER-SEEN reverse directions")
    if results['test']['mrr'] > results['atomic']['mrr']:
        print(f"   ✓ Better MRR than atomic baseline!")
    elif results['test']['mrr'] > 0.5:
        print(f"   ✓ Good MRR (> 0.5)")
    elif results['test']['mrr'] > 0.1:
        print(f"   ⚠ Moderate MRR (0.1-0.5)")
    else:
        print(f"   ✗ Low MRR (< 0.1) - reversal curse present")
    print()

    # MRR comparison
    print(f"4. MRR comparison:")
    print(f"   Train MRR:  {results['train']['mrr']:.4f} (upper bound)")
    print(f"   Atomic MRR: {results['atomic']['mrr']:.4f} (one direction seen)")
    print(f"   Test MRR:   {results['test']['mrr']:.4f} (reversal curse test)")
    mrr_gap = results['train']['mrr'] - results['test']['mrr']
    print(f"   Gap (Train - Test): {mrr_gap:.4f}")
    if mrr_gap < 0.1:
        print(f"   ✓ Minimal reversal curse effect!")
    elif mrr_gap < 0.3:
        print(f"   ⚠ Moderate reversal curse effect")
    else:
        print(f"   ✗ Strong reversal curse effect")
    print()

    # Joint accuracy comparison
    print(f"5. Joint Accuracy (both tokens correct):")
    print(f"   Train:  {results['train']['joint_accuracy']:.4f}")
    print(f"   Atomic: {results['atomic']['joint_accuracy']:.4f}")
    print(f"   Test:   {results['test']['joint_accuracy']:.4f}")
    print()

    # Joint log likelihood comparison
    print(f"6. Joint Log-Likelihood (log P(tok1) + log P(tok2|tok1)):")
    print(f"   Train:  {results['train']['joint_log_likelihood']:.4f}")
    print(f"   Atomic: {results['atomic']['joint_log_likelihood']:.4f}")
    print(f"   Test:   {results['test']['joint_log_likelihood']:.4f}")
    print(f"   (Higher is better; 0 = perfect, more negative = less confident)")
    print()

    # Per-layer analysis if available
    if 'per_layer_mrr' in results['test']:
        print("="*80)
        print("PER-LAYER ANALYSIS (Test Split)")
        print("="*80)
        print()
        print("Layer | MRR (avg) | MRR (tok1) | MRR (tok2) | Joint LL | Joint Acc")
        print("-" * 70)
        n_layers = len(results['test']['per_layer_mrr'])
        for layer in range(n_layers):
            mrr = results['test']['per_layer_mrr'][layer]
            mrr_t1 = results['test']['per_layer_mrr_token1'][layer]
            mrr_t2 = results['test']['per_layer_mrr_token2'][layer]
            joint_ll = results['test']['per_layer_joint_ll'][layer]
            joint = results['test']['per_layer_joint_accuracy'][layer]
            print(f"  {layer:2d}  | {mrr:9.4f} | {mrr_t1:10.4f} | {mrr_t2:10.4f} | {joint_ll:8.4f} | {joint:9.4f}")
        print()

    print("="*90)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Data directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--output', type=str, default=None,
                        help='Optional path to save results JSON')
    parser.add_argument('--per_layer', action='store_true',
                        help='Compute per-layer MRR analysis (slower)')

    args = parser.parse_args()

    results = evaluate_all_splits(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        per_layer=args.per_layer,
    )

    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
