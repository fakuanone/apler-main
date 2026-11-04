"""Shared event-cardinality boundary math for training and generation."""

import math

import torch
import torch.nn.functional as F


def event_count_distribution_tensor(handle_probabilities):
    """Independent-Bernoulli count distribution P(N=0..H)."""

    probabilities = handle_probabilities.float().clamp(0.0, 1.0)
    batch_size, handle_count = probabilities.shape
    distribution = torch.ones(
        batch_size,
        1,
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    for handle_index in range(handle_count):
        probability = probabilities[:, handle_index].unsqueeze(1)
        zero_column = torch.zeros(
            batch_size,
            1,
            dtype=distribution.dtype,
            device=distribution.device,
        )
        absent = torch.cat(
            (distribution * (1.0 - probability), zero_column),
            dim=1,
        )
        present = torch.cat(
            (zero_column, distribution * probability),
            dim=1,
        )
        distribution = absent + present
    return distribution / distribution.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def event_count_target_nll(
    count_distribution,
    target_counts,
    valid_rows=None,
):
    """Normalized NLL of the gold count under P(N=0..H)."""

    distribution = count_distribution.float()
    if distribution.dim() != 2:
        raise ValueError(
            "Count distribution must have shape [B,H+1], "
            f"got {tuple(distribution.shape)}."
        )
    targets = target_counts.to(
        device=distribution.device,
        dtype=torch.long,
    )
    if targets.dim() != 1 or targets.size(0) != distribution.size(0):
        raise ValueError(
            "Target counts must have shape [B] matching the distribution, "
            f"got targets={tuple(targets.shape)} "
            f"distribution={tuple(distribution.shape)}."
        )
    targets = targets.clamp(min=0, max=distribution.size(-1) - 1)
    target_probability = distribution.gather(1, targets.unsqueeze(1)).squeeze(1)
    normalization = max(math.log(float(distribution.size(-1))), 1.0)
    probability_floor = torch.finfo(distribution.dtype).tiny
    losses = -target_probability.clamp_min(probability_floor).log() / normalization

    if valid_rows is not None:
        valid = valid_rows.to(device=distribution.device, dtype=torch.bool)
        if valid.shape != targets.shape:
            raise ValueError(
                "Count-loss valid rows must match target counts, "
                f"got valid={tuple(valid.shape)} targets={tuple(targets.shape)}."
            )
        if not valid.any():
            return distribution.sum() * 0.0
        losses = losses[valid]
    return losses.mean()


def event_count_reliability_tensor(count_distribution):
    """Legacy global normalized count certainty in [0, 1]."""

    distribution = count_distribution.float().clamp_min(0.0)
    if distribution.size(-1) <= 1:
        return torch.ones(
            distribution.shape[:-1],
            dtype=distribution.dtype,
            device=distribution.device,
        )
    entropy = -(
        distribution
        * distribution.clamp_min(1e-12).log()
    ).sum(dim=-1)
    max_entropy = math.log(float(distribution.size(-1)))
    return (1.0 - entropy / max_entropy).clamp(0.0, 1.0)


def conditional_event_continue_probability_tensor(
    count_distribution,
    completed_event_counts,
):
    """Return P(N>k | N>=k) for completed-event counts k."""

    distribution = count_distribution.float().clamp_min(0.0)
    completed = completed_event_counts.to(
        device=distribution.device,
        dtype=torch.long,
    ).clamp(min=0, max=distribution.size(-1) - 1)
    count_values = torch.arange(
        distribution.size(-1),
        device=distribution.device,
    ).unsqueeze(0)
    reached = count_values >= completed.unsqueeze(1)
    continues = count_values > completed.unsqueeze(1)
    reached_mass = (distribution * reached).sum(dim=-1)
    continue_mass = (distribution * continues).sum(dim=-1)
    return torch.where(
        reached_mass > 0.0,
        continue_mass / reached_mass.clamp_min(1e-12),
        torch.zeros_like(reached_mass),
    ).clamp(0.0, 1.0)


def event_boundary_proposal_mask(
    logits,
    continue_token_ids,
    stop_token_ids,
    proposal_vocab_size=None,
):
    """Return rows where Qwen's native top token proposes a boundary branch."""

    if logits.dim() != 2:
        raise ValueError(
            f"Boundary proposal logits must be [N,V], got {tuple(logits.shape)}."
        )
    visible_vocab_size = logits.size(-1)
    if proposal_vocab_size is not None:
        visible_vocab_size = min(
            visible_vocab_size,
            int(proposal_vocab_size),
        )
        if visible_vocab_size <= 0:
            raise ValueError(
                "Boundary proposal vocabulary size must be greater than zero."
            )
    continue_ids = continue_token_ids.to(device=logits.device, dtype=torch.long)
    stop_ids = stop_token_ids.to(device=logits.device, dtype=torch.long)
    if continue_ids.dim() != 2 or stop_ids.dim() != 2:
        raise ValueError("Boundary proposal token ids must both have shape [N,K].")
    if (
        continue_ids.size(0) != logits.size(0)
        or stop_ids.size(0) != logits.size(0)
    ):
        raise ValueError(
            "Boundary proposal token rows must match logits: "
            f"logits={tuple(logits.shape)}, "
            f"continue={tuple(continue_ids.shape)}, "
            f"stop={tuple(stop_ids.shape)}."
        )

    top_ids = logits[:, :visible_vocab_size].float().argmax(dim=-1)
    continue_valid = continue_ids.ge(0) & continue_ids.lt(visible_vocab_size)
    stop_valid = stop_ids.ge(0) & stop_ids.lt(visible_vocab_size)
    continue_match = (
        continue_valid
        & continue_ids.eq(top_ids.unsqueeze(1))
    ).any(dim=-1)
    stop_match = (
        stop_valid
        & stop_ids.eq(top_ids.unsqueeze(1))
    ).any(dim=-1)
    return continue_match | stop_match


def native_event_boundary_statistics(
    logits,
    continue_token_ids,
    stop_token_ids,
    targets,
):
    """Score gold boundaries from the model's unfused continue/stop mass."""

    if logits.dim() != 2:
        raise ValueError(
            f"Native boundary logits must be [N,V], got {tuple(logits.shape)}."
        )
    continue_ids = continue_token_ids.to(device=logits.device, dtype=torch.long)
    stop_ids = stop_token_ids.to(device=logits.device, dtype=torch.long)
    gold_targets = targets.to(device=logits.device, dtype=torch.long)
    if continue_ids.dim() != 2 or stop_ids.dim() != 2:
        raise ValueError("Native boundary token ids must both have shape [N,K].")
    if (
        continue_ids.size(0) != logits.size(0)
        or stop_ids.size(0) != logits.size(0)
        or gold_targets.dim() != 1
        or gold_targets.size(0) != logits.size(0)
    ):
        raise ValueError(
            "Native boundary rows must match logits: "
            f"logits={tuple(logits.shape)}, "
            f"continue={tuple(continue_ids.shape)}, "
            f"stop={tuple(stop_ids.shape)}, "
            f"targets={tuple(gold_targets.shape)}."
        )

    continue_valid = continue_ids.ge(0) & continue_ids.lt(logits.size(-1))
    stop_valid = stop_ids.ge(0) & stop_ids.lt(logits.size(-1))
    valid_rows = (
        continue_valid.any(dim=-1)
        & stop_valid.any(dim=-1)
        & gold_targets.ge(0)
        & gold_targets.le(1)
    )
    safe_continue_ids = continue_ids.clamp(min=0, max=logits.size(-1) - 1)
    safe_stop_ids = stop_ids.clamp(min=0, max=logits.size(-1) - 1)
    logits_fp32 = logits.float()
    continue_values = logits_fp32.gather(1, safe_continue_ids).masked_fill(
        ~continue_valid,
        float("-inf"),
    )
    stop_values = logits_fp32.gather(1, safe_stop_ids).masked_fill(
        ~stop_valid,
        float("-inf"),
    )
    continue_mass = torch.logsumexp(continue_values, dim=-1)
    stop_mass = torch.logsumexp(stop_values, dim=-1)
    native_logits = continue_mass - stop_mass
    valid_rows = (
        valid_rows
        & torch.isfinite(continue_mass)
        & torch.isfinite(stop_mass)
        & torch.isfinite(native_logits)
    )
    safe_logits = torch.where(
        valid_rows,
        native_logits,
        torch.zeros_like(native_logits),
    )
    losses = F.binary_cross_entropy_with_logits(
        safe_logits,
        gold_targets.clamp(0, 1).float(),
        reduction="none",
    )
    losses = torch.where(valid_rows, losses, torch.zeros_like(losses))
    probabilities = torch.where(
        valid_rows,
        torch.sigmoid(safe_logits),
        torch.full_like(safe_logits, float("nan")),
    )
    return losses, probabilities, valid_rows


def mean_event_boundary_loss_per_sample(
    row_losses,
    valid_rows,
    sample_rows,
    batch_size,
    sample_weights=None,
):
    """Average boundary loss within each sample before averaging the batch."""

    if row_losses.dim() != 1 or valid_rows.dim() != 1 or sample_rows.dim() != 1:
        raise ValueError("Boundary loss inputs must all have shape [N].")
    if not (
        row_losses.size(0) == valid_rows.size(0) == sample_rows.size(0)
    ):
        raise ValueError("Boundary loss inputs must contain the same number of rows.")
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("Boundary loss batch_size must be greater than zero.")

    rows = sample_rows.to(device=row_losses.device, dtype=torch.long)
    valid = valid_rows.to(device=row_losses.device, dtype=torch.bool)
    if valid.any():
        valid_sample_rows = rows[valid]
        if (
            valid_sample_rows.min().item() < 0
            or valid_sample_rows.max().item() >= batch_size
        ):
            raise ValueError("Boundary loss sample row is outside the batch.")
    per_sample_sum = torch.zeros(
        batch_size,
        dtype=row_losses.dtype,
        device=row_losses.device,
    )
    per_sample_count = torch.zeros(
        batch_size,
        dtype=row_losses.dtype,
        device=row_losses.device,
    )
    per_sample_sum.scatter_add_(
        0,
        rows,
        row_losses * valid.to(row_losses.dtype),
    )
    per_sample_count.scatter_add_(
        0,
        rows,
        valid.to(row_losses.dtype),
    )
    per_sample_loss = per_sample_sum / per_sample_count.clamp_min(1.0)
    sample_mask = per_sample_count.gt(0)
    if not sample_mask.any():
        return row_losses.sum() * 0.0
    if sample_weights is not None:
        weights = sample_weights.to(
            device=row_losses.device,
            dtype=row_losses.dtype,
        )
        if weights.dim() != 1 or weights.size(0) != batch_size:
            raise ValueError(
                "Boundary sample weights must have shape [B], "
                f"got {tuple(weights.shape)}."
            )
        per_sample_loss = per_sample_loss * weights
    return per_sample_loss[sample_mask].mean()


def blend_event_boundary_logits(
    logits,
    continue_token_ids,
    stop_token_ids,
    handle_continue_probabilities,
    handle_reliabilities,
):
    """Convexly blend continue/stop mass while leaving other logits untouched."""

    if logits.dim() != 2:
        raise ValueError(f"Boundary logits must be [N,V], got {tuple(logits.shape)}.")

    controlled = logits.clone()
    continue_ids = continue_token_ids.to(device=logits.device, dtype=torch.long)
    stop_ids = stop_token_ids.to(device=logits.device, dtype=torch.long)
    continue_valid = continue_ids.ge(0) & continue_ids.lt(logits.size(-1))
    stop_valid = stop_ids.ge(0) & stop_ids.lt(logits.size(-1))
    valid_rows = continue_valid.any(dim=-1) & stop_valid.any(dim=-1)
    if not valid_rows.any():
        empty = torch.full(
            (logits.size(0),),
            float("nan"),
            dtype=torch.float,
            device=logits.device,
        )
        return controlled, empty, empty, valid_rows

    safe_continue_ids = continue_ids.clamp(min=0, max=logits.size(-1) - 1)
    safe_stop_ids = stop_ids.clamp(min=0, max=logits.size(-1) - 1)
    continue_values = controlled.gather(1, safe_continue_ids).float()
    stop_values = controlled.gather(1, safe_stop_ids).float()
    continue_values = continue_values.masked_fill(~continue_valid, float("-inf"))
    stop_values = stop_values.masked_fill(~stop_valid, float("-inf"))
    continue_mass = torch.logsumexp(continue_values, dim=-1)
    stop_mass = torch.logsumexp(stop_values, dim=-1)
    valid_rows = (
        valid_rows
        & torch.isfinite(continue_mass)
        & torch.isfinite(stop_mass)
    )
    qwen_continue = torch.sigmoid(continue_mass - stop_mass)
    reliability = handle_reliabilities.to(
        device=logits.device,
        dtype=torch.float,
    ).clamp(0.0, 1.0)
    handle_continue = handle_continue_probabilities.to(
        device=logits.device,
        dtype=torch.float,
    ).clamp(0.0, 1.0)

    eps = 1e-6
    qwen_safe = qwen_continue.clamp(eps, 1.0 - eps)
    blended_continue = (
        (1.0 - reliability) * qwen_continue
        + reliability * handle_continue
    )

    blended_safe = blended_continue.clamp(eps, 1.0 - eps)
    continue_delta = torch.log(blended_safe) - torch.log(qwen_safe)
    stop_delta = torch.log1p(-blended_safe) - torch.log1p(-qwen_safe)
    continue_delta = torch.where(
        valid_rows,
        continue_delta,
        torch.zeros_like(continue_delta),
    )
    stop_delta = torch.where(
        valid_rows,
        stop_delta,
        torch.zeros_like(stop_delta),
    )

    active_continue = continue_valid & valid_rows.unsqueeze(1)
    active_stop = stop_valid & valid_rows.unsqueeze(1)
    continue_source = (
        continue_delta.to(controlled.dtype).unsqueeze(1)
        * active_continue.to(controlled.dtype)
    )
    stop_source = (
        stop_delta.to(controlled.dtype).unsqueeze(1)
        * active_stop.to(controlled.dtype)
    )
    adjustments = torch.zeros_like(controlled).scatter_add(
        1,
        safe_continue_ids,
        continue_source,
    )
    adjustments = adjustments.scatter_add(
        1,
        safe_stop_ids,
        stop_source,
    )
    controlled = controlled + adjustments
    return controlled, qwen_continue, blended_continue, valid_rows
