"""Training-only, exact vocabulary-space semantic matching."""
import torch
import torch.nn.functional as F
from src.set_utils import minimum_cost_bipartite_assignment


@torch.no_grad()
def vocabulary_distance(instances, references, lm_head, chunk_size=4096):
    if not len(references):
        return instances.new_empty((len(instances), 0), dtype=torch.float32)
    dots = instances.new_zeros((len(instances), len(references)), dtype=torch.float32)
    left_norm = dots.new_zeros(len(instances))
    right_norm = dots.new_zeros(len(references))
    for start in range(0, lm_head.weight.shape[0], chunk_size):
        weight = lm_head.weight[start:start + chunk_size]
        bias = getattr(lm_head, "bias", None)
        bias = bias[start:start + chunk_size] if bias is not None else None
        left = F.linear(instances.to(weight.dtype), weight, bias).float()
        right = F.linear(references.to(weight.dtype), weight, bias).float()
        dots += left @ right.T
        left_norm += left.square().sum(-1)
        right_norm += right.square().sum(-1)
    return 1 - dots / (left_norm.sqrt()[:, None] * right_norm.sqrt()[None, :]).clamp_min(1e-12)


def match_instances(anchor_logits, gold_anchors, instances, references, lm_head, semantic_factor=.2, chunk_size=4096):
    if not gold_anchors:
        return []
    if min(gold_anchors) < 0 or max(gold_anchors) >= anchor_logits.shape[-1]:
        raise ValueError("Every gold instance needs a valid evidence anchor")
    cost = -anchor_logits.detach().float().log_softmax(-1)[:, gold_anchors]
    cost = cost + semantic_factor * vocabulary_distance(instances, references, lm_head, chunk_size)
    return minimum_cost_bipartite_assignment(cost)


def log_count_distribution(existence_logits):
    """Log Poisson-binomial probabilities for counts 0..K, without clipping.

    Grow only reachable counts: padding with -inf would introduce undefined
    gradients for logaddexp(-inf, -inf) at unreachable states.
    """
    logits = existence_logits if existence_logits.dtype == torch.float64 else existence_logits.float()
    log_present, log_absent = F.logsigmoid(logits), F.logsigmoid(-logits)
    distribution = logits.new_zeros(1)
    for yes, no in zip(log_present, log_absent):
        stay, increment = distribution + no, distribution + yes
        distribution = torch.cat((stay[:1], torch.logaddexp(stay[1:], increment[:-1]), increment[-1:]))
    return distribution


def instance_set_loss(anchor_logits, existence_logits, anchors, pairs):
    # M>K is capacity-censored on the instance side; generation retains all M.
    target = torch.zeros_like(existence_logits)
    anchor_loss = anchor_logits.sum() * 0
    if pairs:
        indices, gold_indices = zip(*pairs)
        target[list(indices)] = 1
        anchor_loss = F.cross_entropy(anchor_logits[list(indices)].float(),
                                      torch.tensor([anchors[i] for i in gold_indices], device=anchor_logits.device), reduction="sum")
    existence_loss = F.binary_cross_entropy_with_logits(existence_logits.float(), target.float(), reduction="sum")
    log_distribution = log_count_distribution(existence_logits)
    count_loss = -log_distribution[min(len(anchors), len(existence_logits))]
    return anchor_loss + existence_loss + count_loss, {"anchor": anchor_loss, "existence": existence_loss, "count": count_loss}
