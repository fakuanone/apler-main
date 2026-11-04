"""Paper section 3.4 monotonic interpolation over all four teacher stages."""
import torch
import torch.nn.functional as F


def aligned_teacher(stages, step, steps):
    if stages.shape[-2] != 4 or not 0 <= step < steps:
        raise ValueError("Expected four stages and a valid zero-based student step")
    if steps == 1:
        return stages.mean(dim=-2)
    position = step * 3.0 / (steps - 1)
    lower = int(position)
    upper = min(3, lower + 1)
    alpha = position - lower
    return (1 - alpha) * stages[..., lower, :] + alpha * stages[..., upper, :]


def process_loss(student_layers, teacher_layers, step, steps):
    if len(student_layers) != len(teacher_layers) or not student_layers:
        raise ValueError("Teacher and student must expose identical specified layers")
    losses = []
    for student, teacher in zip(student_layers, teacher_layers):
        reference = aligned_teacher(teacher.detach(), step, steps)
        losses.append((1 - F.cosine_similarity(student.float(), reference.float(), dim=-1)).mean())
    return torch.stack(losses).mean()
