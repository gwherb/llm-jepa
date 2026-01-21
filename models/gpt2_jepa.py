import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Model, GPT2LMHeadModel
from typing import Optional, Tuple, Dict
import os
from .gpt2_jepa_config import JEPAConfig, GPT2JEPAConfig

class GPT2WithJEPA(nn.Module):

    def __init__(
            self,
            base_model: GPT2LMHeadModel,
            jepa_config: JEPAConfig,
    ):
        super().__init__()

        self.base_model = base_model
        self.config = base_model.config
        self.jepa_config = jepa_config

        self.hidden_size = self.config.hidden_size

    def _create_block_causal_mask(
        self,
        seq_len: int,
        k_pred: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create block-diagonal causal attention mask for JEPA.

        The mask prevents the forward sequence (with PRED tokens) and reverse sequence
        from attending to each other, while maintaining causal attention within each block.

        Structure:
        - Forward sequence: [e1, rel, mask, e2, PRED_0, ..., PRED_{k-1}]  (seq_len + k tokens)
        - Reverse sequence: [e2, rel_inv, mask, e1]                        (seq_len tokens)

        Args:
            seq_len: Base sequence length (same for both forward and reverse, e.g., 6)
            k_pred: Number of predictor tokens
            batch_size: Batch size
            device: Device for the tensor

        Returns:
            Attention mask of shape (batch_size, 1, total_len, total_len)
            where total_len = (seq_len + k_pred) + seq_len
        """
        forward_len = seq_len + k_pred
        reverse_len = seq_len
        total_len = forward_len + reverse_len

        # Initialize mask with -inf (no attention allowed)
        mask = torch.full((batch_size, 1, total_len, total_len), float('-inf'), device=device)

        # Block 1: Forward sequence with PRED tokens (causal)
        forward_mask = torch.triu(torch.ones(forward_len, forward_len, device=device), diagonal=1)
        forward_mask = torch.where(forward_mask == 1, torch.tensor(float('-inf'), device=device), torch.tensor(0.0, device=device))
        mask[:, :, :forward_len, :forward_len] = forward_mask.unsqueeze(0).unsqueeze(0)

        # Block 2: Reverse sequence (causal)
        reverse_mask = torch.triu(torch.ones(reverse_len, reverse_len, device=device), diagonal=1)
        reverse_mask = torch.where(reverse_mask == 1, torch.tensor(float('-inf'), device=device), torch.tensor(0.0, device=device))
        mask[:, :, forward_len:, forward_len:] = reverse_mask.unsqueeze(0).unsqueeze(0)

        # Blocks do NOT attend to each other (already set to -inf)

        return mask

    def _extract_embedding(self, hidden_states: torch.Tensor, position: int = -1) -> torch.Tensor:
        """
        Extract embedding from hidden states at a specific position.

        Args:
            hidden_states: (batch_size, seq_len, hidden_size)
            position: Position to extract from (-1 for last token)

        Returns:
            (batch_size, hidden_size)
        """
        return hidden_states[:, position, :]

    def _compute_jepa_loss(self, pred_emb: torch.Tensor, target_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute JEPA loss between predicted and target embeddings.

        Args:
            pred_emb: Predicted embedding (batch_size, hidden_size)
            target_emb: Target embedding (batch_size, hidden_size)

        Returns:
            Loss value (scalar or per-sample depending on reduction)
        """
        if self.jepa_config.distance_metric == 'cosine':
            # 1 - cosine_similarity (minimize distance)
            cos_sim = F.cosine_similarity(pred_emb, target_emb, dim=-1)
            loss = 1 - cos_sim
        elif self.jepa_config.distance_metric == 'l2':
            loss = torch.norm(pred_emb - target_emb, p=2, dim=-1)
        elif self.jepa_config.distance_metric == 'mse':
            loss = F.mse_loss(pred_emb, target_emb, reduction='none').mean(dim=-1)
        else:
            raise ValueError(f"Unknown distance metric: {self.jepa_config.distance_metric}")

        return loss.mean()  # Average over batch

    def forward_jepa(
        self,
        forward_input_ids: torch.Tensor,
        forward_labels: torch.Tensor,
        reverse_input_ids: Optional[torch.Tensor] = None,
        compute_jepa: Optional[torch.Tensor] = None,
        pred_token_id: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with JEPA loss computation.

        Args:
            forward_input_ids: Forward sequence tokens (batch_size, seq_len)
                              Format: [e1_first, e1_last, relation, mask, e2_first, e2_last]
            forward_labels: Labels for NTP loss (batch_size, seq_len)
            reverse_input_ids: Reverse sequence tokens (batch_size, seq_len), optional
                              Format: [e2_first, e2_last, relation_inv, mask, e1_first, e1_last]
            compute_jepa: Boolean mask indicating which samples to compute JEPA loss for (batch_size,)
                         True for type="train", False for type="atomic"
            pred_token_id: Token ID for the <PRED> token

        Returns:
            Dictionary containing:
                - 'loss': Combined loss (gamma * NTP + lambda * JEPA)
                - 'ntp_loss': Next token prediction loss
                - 'jepa_loss': JEPA embedding loss (0 if not computed)
                - 'logits': Model logits
        """
        batch_size, seq_len = forward_input_ids.shape
        device = forward_input_ids.device

        # 1. Compute NTP loss on forward sequence
        forward_outputs = self.base_model(
            input_ids=forward_input_ids,
            labels=forward_labels,
        )
        ntp_loss = forward_outputs.loss
        logits = forward_outputs.logits

        # 2. Compute JEPA loss if applicable
        jepa_loss = torch.tensor(0.0, device=device)

        if (self.jepa_config.use_jepa and
            reverse_input_ids is not None and
            compute_jepa is not None and
            compute_jepa.any()):

            # Apply loss dropout
            if self.jepa_config.loss_dropout > 0:
                dropout_mask = torch.rand(batch_size, device=device) > self.jepa_config.loss_dropout
                compute_jepa = compute_jepa & dropout_mask

            if compute_jepa.any():
                # Get indices where JEPA should be computed
                jepa_indices = compute_jepa.nonzero(as_tuple=True)[0]

                # Select relevant samples
                forward_ids_jepa = forward_input_ids[jepa_indices]
                reverse_ids_jepa = reverse_input_ids[jepa_indices]
                jepa_batch_size = forward_ids_jepa.shape[0]

                k = self.jepa_config.k_pred_tok

                # Add PRED tokens to forward sequence if k > 0
                if k > 0 and pred_token_id is not None:
                    pred_tokens = torch.full((jepa_batch_size, k), pred_token_id, device=device, dtype=torch.long)
                    forward_with_pred = torch.cat([forward_ids_jepa, pred_tokens], dim=1)
                else:
                    forward_with_pred = forward_ids_jepa

                # Concatenate forward (with PRED) and reverse sequences
                combined_input_ids = torch.cat([forward_with_pred, reverse_ids_jepa], dim=1)

                # Create block-causal attention mask
                attention_mask = self._create_block_causal_mask(
                    seq_len=seq_len,
                    k_pred=k,
                    batch_size=jepa_batch_size,
                    device=device,
                )

                # Forward pass with custom attention mask
                outputs = self.base_model.transformer(
                    input_ids=combined_input_ids,
                    attention_mask=attention_mask,
                )
                hidden_states = outputs.last_hidden_state  # (jepa_batch_size, total_len, hidden_size)

                # Extract embeddings
                # Predicted embedding: last token of forward sequence (last PRED token or last regular token)
                forward_end_idx = seq_len + k - 1
                pred_emb = hidden_states[:, forward_end_idx, :]  # (jepa_batch_size, hidden_size)

                # Target embedding: last token of reverse sequence
                target_emb = hidden_states[:, -1, :]  # (jepa_batch_size, hidden_size)

                # Compute JEPA loss
                jepa_loss = self._compute_jepa_loss(pred_emb, target_emb)

        # 3. Combine losses
        total_loss = self.jepa_config.gamma_ntp * ntp_loss + self.jepa_config.lambda_jepa * jepa_loss

        return {
            'loss': total_loss,
            'ntp_loss': ntp_loss,
            'jepa_loss': jepa_loss,
            'logits': logits,
        }

    def forward(self, *args, **kwargs):
        """Default forward pass delegates to forward_jepa."""
        return self.forward_jepa(*args, **kwargs)

    @staticmethod
    def create_reverse_sequence(
        forward_input_ids: torch.Tensor,
        first_relation_token_id: int,
    ) -> torch.Tensor:
        """
        Create reverse sequence from forward sequence by swapping entities and inverting relation.

        Forward format:  [e1_first, e1_last, relation, mask, e2_first, e2_last]
        Reverse format:  [e2_first, e2_last, relation_inv, mask, e1_first, e1_last]

        Relation inversion logic:
        - Relations come in pairs: <r_i> and <r_i_inv>
        - If first_relation_token_id is odd, then <r_i> is odd and <r_i_inv> is even
        - If first_relation_token_id is even, then <r_i> is even and <r_i_inv> is odd

        Args:
            forward_input_ids: (batch_size, seq_len) where seq_len should be 6
                              Format: [e1_first, e1_last, relation, mask, e2_first, e2_last]
            first_relation_token_id: Token ID of the first relation token (e.g., <r_0>)
                                    Used to determine the parity pattern

        Returns:
            reverse_input_ids: (batch_size, seq_len)
                              Format: [e2_first, e2_last, relation_inv, mask, e1_first, e1_last]
        """
        batch_size, seq_len = forward_input_ids.shape
        assert seq_len == 6, f"Expected seq_len=6, got {seq_len}"

        # Extract components
        e1_first = forward_input_ids[:, 0:1]   # (batch_size, 1)
        e1_last = forward_input_ids[:, 1:2]    # (batch_size, 1)
        relation = forward_input_ids[:, 2:3]   # (batch_size, 1)
        mask = forward_input_ids[:, 3:4]       # (batch_size, 1)
        e2_first = forward_input_ids[:, 4:5]   # (batch_size, 1)
        e2_last = forward_input_ids[:, 5:6]    # (batch_size, 1)

        # Invert relation token
        # Determine if first_relation_token_id is odd or even
        first_is_odd = (first_relation_token_id % 2) == 1

        # If first is odd: odd tokens are forward, even tokens are inverse
        # If first is even: even tokens are forward, odd tokens are inverse
        relation_is_odd = (relation % 2) == 1

        if first_is_odd:
            # Forward relations are odd, inverse relations are even
            # If current relation is odd (forward), add 1 to get inverse
            # If current relation is even (inverse), subtract 1 to get forward
            relation_inv = torch.where(
                relation_is_odd,
                relation + 1,  # odd -> even (forward -> inverse)
                relation - 1,  # even -> odd (inverse -> forward)
            )
        else:
            # Forward relations are even, inverse relations are odd
            # If current relation is even (forward), add 1 to get inverse
            # If current relation is odd (inverse), subtract 1 to get forward
            relation_inv = torch.where(
                relation_is_odd,
                relation - 1,  # odd -> even (inverse -> forward)
                relation + 1,  # even -> odd (forward -> inverse)
            )

        # Construct reverse sequence: [e2_first, e2_last, relation_inv, mask, e1_first, e1_last]
        reverse_input_ids = torch.cat([e2_first, e2_last, relation_inv, mask, e1_first, e1_last], dim=1)

        return reverse_input_ids


def train_jepa(
    model: GPT2WithJEPA,
    train_dataloader,
    optimizer,
    num_epochs: Optional[int],
    device: torch.device,
    pred_token_id: int,
    first_relation_token_id: int,
    max_steps: Optional[int] = None,
    eval_dataloader=None,
    scheduler=None,
    log_interval: int = 100,
    save_steps: int = 10000,
    save_step_dense: Optional[int] = None,
    save_step_dense_interval: Optional[int] = None,
    eval_steps: int = 5000,
    save_path: Optional[str] = None,
):
    """
    Training loop for GPT2 with JEPA.

    This function handles the complete training loop including:
    - Automatic reverse sequence generation for JEPA loss
    - Selective JEPA loss computation based on sample type
    - Combined NTP + JEPA loss optimization
    - Periodic evaluation and logging
    - Step-based or epoch-based training

    Args:
        model: GPT2WithJEPA model instance
        train_dataloader: DataLoader yielding batches with:
            - 'input_ids': (batch_size, seq_len) forward sequences
            - 'labels': (batch_size, seq_len) labels for NTP loss
            - 'type': List of 'train' or 'atomic' indicating JEPA eligibility
        optimizer: Optimizer instance
        num_epochs: Number of training epochs (if max_steps not specified)
        device: Device to train on
        pred_token_id: Token ID for <PRED> token
        first_relation_token_id: Token ID of first relation (e.g., <r_0> = 16001)
        max_steps: Maximum number of training steps (overrides num_epochs)
        eval_dataloader: Optional evaluation dataloader
        scheduler: Optional learning rate scheduler
        log_interval: Steps between logging
        save_steps: Steps between checkpoint saves
        save_step_dense: Save more frequently until this step
        save_step_dense_interval: Interval for dense checkpoint saving
        eval_steps: Steps between evaluation runs
        save_path: Optional path to save model checkpoints

    Returns:
        Dictionary with training metrics history
    """
    model.train()
    model.to(device)

    # Determine training mode: step-based or epoch-based
    if max_steps is not None:
        training_mode = 'steps'
        total_steps = max_steps
        # Calculate approximate epochs for display
        approx_epochs = max_steps / len(train_dataloader)
    else:
        training_mode = 'epochs'
        total_steps = num_epochs * len(train_dataloader)
        approx_epochs = num_epochs

    global_step = 0
    training_stats = {
        'step': [],
        'total_loss': [],
        'ntp_loss': [],
        'jepa_loss': [],
    }

    print("Starting LLM-JEPA Training...")
    print(f"Config: lambda={model.jepa_config.lambda_jepa}, gamma={model.jepa_config.gamma_ntp}, "
          f"k={model.jepa_config.k_pred_tok}, metric={model.jepa_config.distance_metric}")
    print(f"Loss dropout: {model.jepa_config.loss_dropout}")
    print(f"Training mode: {training_mode} ({'max_steps=' + str(max_steps) if training_mode == 'steps' else 'num_epochs=' + str(num_epochs)})")
    print(f"Approximate epochs: {approx_epochs:.1f}\n")

    epoch = 0
    finished = False

    while not finished:
        if training_mode == 'epochs' and epoch >= num_epochs:
            break
        epoch_loss = 0.0
        epoch_ntp_loss = 0.0
        epoch_jepa_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(train_dataloader):
            # Check if we've reached max_steps
            if training_mode == 'steps' and global_step >= max_steps:
                finished = True
                break

            # Move batch to device
            forward_input_ids = batch['input_ids'].to(device)
            forward_labels = batch['labels'].to(device)
            batch_types = batch['type']  # List of 'train' or 'atomic'

            # Create compute_jepa mask (True for 'train', False for 'atomic')
            compute_jepa = torch.tensor(
                [t == 'train' for t in batch_types],
                dtype=torch.bool,
                device=device
            )

            # Generate reverse sequences automatically
            reverse_input_ids = None
            if compute_jepa.any():  # Only generate if needed
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            # Accumulate stats
            epoch_loss += loss.item()
            epoch_ntp_loss += ntp_loss.item()
            epoch_jepa_loss += jepa_loss.item() if isinstance(jepa_loss, torch.Tensor) else jepa_loss
            num_batches += 1

            # Logging
            if global_step % log_interval == 0:
                jepa_val = jepa_loss.item() if isinstance(jepa_loss, torch.Tensor) else jepa_loss
                current_epoch = epoch + 1
                print(f"[Epoch {current_epoch}] Step {global_step} | "
                      f"Loss: {loss.item():.4f} | NTP: {ntp_loss.item():.4f} | JEPA: {jepa_val:.4f}")

            # Evaluation
            if eval_dataloader is not None and global_step > 0 and global_step % eval_steps == 0:
                print(f"\nRunning evaluation at step {global_step}...")
                eval_metrics = evaluate_jepa(
                    model, eval_dataloader, device, pred_token_id, first_relation_token_id,
                    compute_mrr=False  # Skip MRR during training for speed
                )
                print(f"Eval Loss: {eval_metrics['loss']:.4f} | "
                      f"NTP: {eval_metrics['ntp_loss']:.4f} | "
                      f"JEPA: {eval_metrics['jepa_loss']:.4f}\n")
                model.train()

            # Checkpoint saving with dense saving logic
            if save_path is not None and global_step > 0:
                should_save = False

                # Dense saving: save every save_step_dense_interval until save_step_dense
                if (save_step_dense is not None and
                    save_step_dense_interval is not None and
                    global_step <= save_step_dense):
                    if global_step % save_step_dense_interval == 0:
                        should_save = True

                # Regular saving: save every save_steps after save_step_dense
                elif global_step % save_steps == 0:
                    should_save = True

                if should_save:
                    checkpoint_path = f"{save_path}/checkpoints/checkpoint_step_{global_step}.pt"
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                    torch.save({
                        'step': global_step,
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'jepa_config': model.jepa_config,
                        'training_stats': training_stats,
                    }, checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")

            global_step += 1

        # Epoch summary (only if we completed batches)
        if num_batches > 0:
            avg_loss = epoch_loss / num_batches
            avg_ntp = epoch_ntp_loss / num_batches
            avg_jepa = epoch_jepa_loss / num_batches

            training_stats['step'].append(global_step)
            training_stats['total_loss'].append(avg_loss)
            training_stats['ntp_loss'].append(avg_ntp)
            training_stats['jepa_loss'].append(avg_jepa)

            print(f"\n{'='*70}")
            print(f"Epoch {epoch+1} Summary (Step {global_step}):")
            print(f"  Train Loss: {avg_loss:.4f} | NTP: {avg_ntp:.4f} | JEPA: {avg_jepa:.4f}")
            print(f"{'='*70}\n")

        epoch += 1

    print("Training completed!")
    print(f"Final step: {global_step}")
    return training_stats


def evaluate_jepa(
    model: GPT2WithJEPA,
    eval_dataloader,
    device: torch.device,
    pred_token_id: int,
    first_relation_token_id: int,
    compute_mrr: bool = True,
    compute_per_layer: bool = False,
):
    """
    Evaluate GPT2 with JEPA.

    Args:
        model: GPT2WithJEPA model instance
        eval_dataloader: Evaluation dataloader
        device: Device to evaluate on
        pred_token_id: Token ID for <PRED> token
        first_relation_token_id: Token ID of first relation
        compute_mrr: Whether to compute MRR (expensive, skip during training)
        compute_per_layer: Whether to compute per-layer MRR (very expensive)

    Returns:
        Dictionary with evaluation metrics (including MRR if compute_mrr=True)
    """
    model.eval()
    total_loss = 0.0
    total_ntp_loss = 0.0
    total_jepa_loss = 0.0
    num_batches = 0

    # MRR tracking (final layer)
    total_reciprocal_rank = 0.0
    total_reciprocal_rank_token1 = 0.0
    total_reciprocal_rank_token2 = 0.0
    total_joint_log_likelihood = 0.0  # Joint log likelihood of entity
    total_samples = 0
    joint_correct = 0  # Both tokens rank 1

    # Per-layer tracking
    n_layers = model.config.n_layer
    if compute_per_layer:
        per_layer_rr = {layer: 0.0 for layer in range(n_layers)}
        per_layer_rr_token1 = {layer: 0.0 for layer in range(n_layers)}
        per_layer_rr_token2 = {layer: 0.0 for layer in range(n_layers)}
        per_layer_joint_ll = {layer: 0.0 for layer in range(n_layers)}
        per_layer_joint_correct = {layer: 0 for layer in range(n_layers)}

    with torch.no_grad():
        for batch in eval_dataloader:
            forward_input_ids = batch['input_ids'].to(device)
            forward_labels = batch['labels'].to(device)
            batch_types = batch['type']

            compute_jepa_mask = torch.tensor(
                [t == 'train' for t in batch_types],
                dtype=torch.bool,
                device=device
            )

            reverse_input_ids = None
            if compute_jepa_mask.any():
                reverse_input_ids = model.create_reverse_sequence(
                    forward_input_ids,
                    first_relation_token_id
                )

            outputs = model.forward_jepa(
                forward_input_ids=forward_input_ids,
                forward_labels=forward_labels,
                reverse_input_ids=reverse_input_ids,
                compute_jepa=compute_jepa_mask,
                pred_token_id=pred_token_id,
            )

            total_loss += outputs['loss'].item()
            total_ntp_loss += outputs['ntp_loss'].item()
            total_jepa_loss += outputs['jepa_loss'].item() if isinstance(outputs['jepa_loss'], torch.Tensor) else outputs['jepa_loss']
            num_batches += 1

            # Compute MRR only if requested (expensive operation)
            if compute_mrr:
                logits = outputs['logits']  # (batch_size, seq_len, vocab_size)
                batch_size = forward_input_ids.shape[0]

                # For each sample, compute metrics for both target tokens
                # Labels format: [-100, -100, -100, -100, e2_first, e2_last]
                # Positions 4 and 5 have valid labels
                for b in range(batch_size):
                    # Token 1 (e2_first) - predicted from position 3
                    if forward_labels[b, 4] != -100:
                        token1_logits = logits[b, 3, :]
                        target1 = forward_labels[b, 4].item()
                        sorted1 = torch.argsort(token1_logits, descending=True)
                        rank1 = (sorted1 == target1).nonzero(as_tuple=True)[0].item() + 1
                        rr1 = 1.0 / rank1
                        total_reciprocal_rank_token1 += rr1

                        # Token 2 (e2_last) - predicted from position 4
                        token2_logits = logits[b, 4, :]
                        target2 = forward_labels[b, 5].item()
                        sorted2 = torch.argsort(token2_logits, descending=True)
                        rank2 = (sorted2 == target2).nonzero(as_tuple=True)[0].item() + 1
                        rr2 = 1.0 / rank2
                        total_reciprocal_rank_token2 += rr2

                        # Averaged MRR
                        total_reciprocal_rank += (rr1 + rr2) / 2

                        # Joint accuracy (both rank 1)
                        if rank1 == 1 and rank2 == 1:
                            joint_correct += 1

                        # Joint log likelihood: log P(tok1) + log P(tok2|tok1)
                        log_probs1 = torch.log_softmax(token1_logits, dim=-1)
                        log_probs2 = torch.log_softmax(token2_logits, dim=-1)
                        joint_ll = log_probs1[target1].item() + log_probs2[target2].item()
                        total_joint_log_likelihood += joint_ll

                        total_samples += 1

                # Per-layer analysis
                if compute_per_layer:
                    # Get hidden states from all layers
                    layer_outputs = model.base_model.transformer(
                        input_ids=forward_input_ids,
                        output_hidden_states=True,
                    )
                    all_hidden_states = layer_outputs.hidden_states  # Tuple of (batch, seq, hidden)

                    # Get the LM head
                    lm_head = model.base_model.lm_head

                    # For each layer (skip embedding layer at index 0)
                    for layer_idx in range(n_layers):
                        hidden = all_hidden_states[layer_idx + 1]  # +1 to skip embedding

                        # Project to vocab
                        layer_logits = lm_head(hidden)  # (batch, seq, vocab)

                        for b in range(batch_size):
                            if forward_labels[b, 4] != -100:
                                # Token 1
                                t1_logits = layer_logits[b, 3, :]
                                target1 = forward_labels[b, 4].item()
                                sorted1 = torch.argsort(t1_logits, descending=True)
                                rank1 = (sorted1 == target1).nonzero(as_tuple=True)[0].item() + 1
                                rr1 = 1.0 / rank1
                                per_layer_rr_token1[layer_idx] += rr1

                                # Token 2
                                t2_logits = layer_logits[b, 4, :]
                                target2 = forward_labels[b, 5].item()
                                sorted2 = torch.argsort(t2_logits, descending=True)
                                rank2 = (sorted2 == target2).nonzero(as_tuple=True)[0].item() + 1
                                rr2 = 1.0 / rank2
                                per_layer_rr_token2[layer_idx] += rr2

                                # Average
                                per_layer_rr[layer_idx] += (rr1 + rr2) / 2

                                # Joint
                                if rank1 == 1 and rank2 == 1:
                                    per_layer_joint_correct[layer_idx] += 1

                                # Joint log likelihood for this layer
                                log_p1 = torch.log_softmax(t1_logits, dim=-1)
                                log_p2 = torch.log_softmax(t2_logits, dim=-1)
                                joint_ll = log_p1[target1].item() + log_p2[target2].item()
                                per_layer_joint_ll[layer_idx] += joint_ll

    model.train()

    results = {
        'loss': total_loss / num_batches,
        'ntp_loss': total_ntp_loss / num_batches,
        'jepa_loss': total_jepa_loss / num_batches,
    }

    if compute_mrr and total_samples > 0:
        results['mrr'] = total_reciprocal_rank / total_samples
        results['mrr_token1'] = total_reciprocal_rank_token1 / total_samples
        results['mrr_token2'] = total_reciprocal_rank_token2 / total_samples
        results['joint_log_likelihood'] = total_joint_log_likelihood / total_samples
        results['joint_accuracy'] = joint_correct / total_samples
        results['num_samples'] = total_samples

        if compute_per_layer:
            results['per_layer_mrr'] = {layer: per_layer_rr[layer] / total_samples for layer in range(n_layers)}
            results['per_layer_mrr_token1'] = {layer: per_layer_rr_token1[layer] / total_samples for layer in range(n_layers)}
            results['per_layer_mrr_token2'] = {layer: per_layer_rr_token2[layer] / total_samples for layer in range(n_layers)}
            results['per_layer_joint_ll'] = {layer: per_layer_joint_ll[layer] / total_samples for layer in range(n_layers)}
            results['per_layer_joint_accuracy'] = {layer: per_layer_joint_correct[layer] / total_samples for layer in range(n_layers)}

    return results
