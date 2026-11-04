"""Exact target-token mean CE with bounded vocabulary activation storage."""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def chunked_generation_loss(hidden, labels, lm_head, chunk_size):
    if chunk_size < 1 or labels.numel() == 0 or hidden.shape[:-1] != labels.shape:
        raise ValueError("Expected aligned nonempty hidden/label sequences and a positive chunk size")

    def project_and_score(states, targets):
        logits = lm_head(states.to(lm_head.weight.dtype)).float()
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum")

    loss = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, labels.shape[1], chunk_size):
        states, targets = hidden[:, start:start + chunk_size], labels[:, start:start + chunk_size]
        # Checkpoint BOTH projection and CE. Checkpointing just the projection
        # would still retain every chunk's full-vocabulary log-softmax for CE.
        if torch.is_grad_enabled():
            block = checkpoint(project_and_score, states, targets, use_reentrant=False)
        else:
            block = project_and_score(states, targets)
        loss = loss + block
    return loss / labels.numel()
