"""Small offline causal-attention backbone, exclusively for pipeline smoke tests.

This is randomly initialized, has no LoRA/pretraining, and produces no meaningful
extraction scores. It must never be reported as a Qwen or paper experiment.
"""
import hashlib
from types import SimpleNamespace
import torch
from torch import nn


class TinyLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q, self.k, self.v, self.out = (nn.Linear(dim, dim) for _ in range(4))

    def forward(self, x, past=None):
        q, k, v = self.q(x), self.k(x), self.v(x)
        previous = 0 if past is None else past[0].shape[1]
        if past is not None: k, v = torch.cat((past[0], k), 1), torch.cat((past[1], v), 1)
        scores = q @ k.transpose(-1, -2)/(x.shape[-1]**.5)
        mask = torch.arange(k.shape[1], device=x.device)[None, :] > previous + torch.arange(x.shape[1], device=x.device)[:, None]
        h = x + self.out(scores.masked_fill(mask, float('-inf')).softmax(-1) @ v)
        return h, (k, v)


class TinyBackbone(nn.Module):
    def __init__(self, vocab=32):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, 16)
        self.layers = nn.ModuleList([TinyLayer(16) for _ in range(4)])
        self.norm = nn.LayerNorm(16)

    def forward(self, input_ids=None, inputs_embeds=None, past_key_values=None, **kwargs):
        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        cache = []
        for i, layer in enumerate(self.layers):
            x, state = layer(x, None if past_key_values is None else past_key_values[i]); cache.append(state)
        return SimpleNamespace(last_hidden_state=self.norm(x), past_key_values=tuple(cache))


class TinyLM(nn.Module):
    def __init__(self, vocab=32, context=512):
        super().__init__()
        self.model = TinyBackbone(vocab); self.lm_head = nn.Linear(16, vocab, bias=False)
        self.config = SimpleNamespace(hidden_size=16, max_position_embeddings=context)

    def get_input_embeddings(self): return self.model.embed_tokens
    def get_output_embeddings(self): return self.lm_head


class SmokeTokenizer:
    """Fixed coarse hashing with offsets. Tests the pipeline, not token accuracy."""
    eos_token_id = 1
    chat_template = None
    def __call__(self, text, **kwargs):
        offsets = [(i, min(i+16, len(text))) for i in range(0, len(text), 16)]
        ids = [2 + int.from_bytes(hashlib.sha256(text[a:b].encode()).digest()[:4], 'big') % 254 for a, b in offsets]
        return {"input_ids": ids, "offset_mapping": offsets}

    def encode(self, text, **kwargs): return self(text)["input_ids"]
    def decode(self, ids, **kwargs): return " ".join(f"<smoke:{i}>" for i in ids)


def load_smoke(config, statistics):
    from src.model import APLER
    torch.set_num_threads(1)
    return APLER(TinyLM(vocab=256, context=8192), config, statistics), SmokeTokenizer()
