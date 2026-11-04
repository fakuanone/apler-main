"""APLER paper-v1. One document per micro-batch; shared causal backbone."""
import math
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from src.dataset_spec import METHOD_VERSION, write_json
from src.event_matching import match_instances, instance_set_loss
from src.latent_budget import gold_budget_probability
from src.process_distillation import process_loss
from src.generation_loss import chunked_generation_loss
from src.checkpoints import checkpoint_metadata


class APLER(nn.Module):
    def __init__(self, causal_lm, config, statistics):
        super().__init__()
        self.codi = causal_lm
        self.paper_config, self.statistics = dict(config), dict(statistics)
        self.dim = causal_lm.config.hidden_size
        self.capacity = int(statistics["instance_capacity"])
        self.max_steps = int(config["max_steps"])
        self.layers = list(config["distill_layers"])
        if not self.layers or min(self.layers) < 1 or max(self.layers) > len(self.backbone.layers):
            raise ValueError("Specified one-based distillation layers do not exist")
        d = self.dim
        self.document_gate = nn.Linear(2*d, d)
        self.instance_queries = nn.Embedding(self.capacity, d)
        self.sentence_projection = nn.Linear(d, d, bias=False)
        self.instance_norm = nn.LayerNorm(d)
        self.existence = nn.Linear(d, 1)
        self.budget_head = nn.Linear(d, self.max_steps)
        self.budget_embedding = nn.Embedding(self.max_steps, d)
        self.budget_count = nn.Sequential(nn.Linear(2*d, d), nn.Tanh(), nn.Linear(d, 1))
        self.route_q = nn.Linear(d, d, bias=False)
        self.route_k = nn.Linear(d, d, bias=False)
        self.route_v = nn.Linear(d, d, bias=False)
        self.route_gate = nn.Linear(2*d, d)
        self.bridge_projection = nn.Linear(d, d, bias=False)
        self.bridge_gate = nn.Linear(2*d, d)

    @property
    def base(self):
        return self.codi.get_base_model() if hasattr(self.codi, "get_base_model") else self.codi

    @property
    def backbone(self):
        return self.base.model

    @property
    def lm_head(self):
        return self.base.get_output_embeddings()

    @property
    def device(self):
        return self.instance_queries.weight.device

    def ids(self, values):
        return torch.tensor([values], device=self.device, dtype=torch.long)

    def run_backbone(self, *, capture=False, ranges=None, **kwargs):
        captured, handles = {}, []
        if capture:
            def hook_for(index):
                def hook(module, inputs, output):
                    h = output[0] if isinstance(output, tuple) else output
                    captured[index] = (torch.stack([h[:, a:b].float().mean(1) for a, b in ranges], dim=1)
                                       if ranges is not None else h[:, -1].float())
                return hook
            for index in self.layers:
                handles.append(self.backbone.layers[index-1].register_forward_hook(hook_for(index)))
        try:
            output = self.backbone(return_dict=True, output_hidden_states=False, **kwargs)
        finally:
            for handle in handles: handle.remove()
        return output, [captured[i] for i in self.layers] if capture else []

    def encode_document(self, item):
        output, _ = self.run_backbone(input_ids=self.ids(item["input_ids"]), use_cache=True)
        states = output.last_hidden_state[0].float()
        if not item["document_positions"]: raise ValueError("No document content tokens")
        last, pooled = states[-1], states[item["document_positions"]].mean(0)
        gate = self.document_gate(torch.cat((last, pooled))).sigmoid()
        summary = gate*last + (1-gate)*pooled
        return output.past_key_values, summary, states[item["sentence_positions"]]

    def budget(self, summary):
        logits = self.budget_head(summary)
        probabilities = logits.float().softmax(-1)
        expected = (probabilities * torch.arange(1, self.max_steps+1, device=self.device)).sum()
        proposal = int(torch.floor(expected.detach()+.5).clamp(1, self.max_steps).item())
        return logits, probabilities, expected, proposal

    def plan_instances(self, summary, sentences, probabilities):
        query = F.normalize(self.instance_queries.weight, dim=-1) + F.normalize(summary, dim=-1)
        normalized = query / query.norm(dim=-1, keepdim=True).clamp_min(.25)
        keys = F.normalize(self.sentence_projection(sentences), dim=-1)
        attention = (normalized @ keys.T * math.sqrt(self.dim)).softmax(-1)
        instances = self.instance_norm(query + attention @ sentences)
        anchors = F.normalize(instances, dim=-1) @ keys.T * math.sqrt(self.dim)
        budget_context = (probabilities.detach() @ self.budget_embedding.weight.detach()).expand(self.capacity, -1)
        exists = self.existence(instances).squeeze(-1) + self.budget_count(torch.cat((instances, budget_context), -1)).squeeze(-1)
        return instances, anchors, exists

    def choose_active(self, anchors, exists, *, gold_anchors=None, pairs=None, progress=1.0):
        use_gold = self.training and gold_anchors is not None and len(gold_anchors) > 0 and torch.rand((), device=self.device).item() < gold_budget_probability(progress)
        if use_gold:
            active = [i for i, _ in pairs]
            order = {i: gold_anchors[j] for i, j in pairs}
        else:
            count = int(torch.floor(exists.sigmoid().sum().detach()+.5).clamp(1, self.capacity).item())
            active = exists.detach().argsort(descending=True)[:count].tolist()
            order = {i: int(anchors[i].argmax().item()) for i in active}
        return sorted(active, key=lambda i: (order[i], i))

    def route(self, latent, memory):
        weights = (self.route_q(latent) @ self.route_k(memory).T / math.sqrt(self.dim)).softmax(-1)
        context = weights @ self.route_v(memory)
        gate = self.route_gate(torch.cat((latent, context))).sigmoid()
        return latent + self.paper_config["routing_factor"]*gate*context

    def bridge(self, latent, memory):
        base = latent.expand(len(memory), -1)
        gate = self.bridge_gate(torch.cat((base, memory), -1)).sigmoid()
        return base + self.paper_config["bridge_factor"]*gate*self.bridge_projection(memory)

    @torch.no_grad()
    def teacher_targets(self, item):
        output, stages = self.run_backbone(input_ids=self.ids(item["teacher_ids"]), capture=True,
                                          ranges=item["stage_ranges"], use_cache=False)
        del output
        references = []
        for ids in item["event_reference_ids"]:
            output, _ = self.run_backbone(input_ids=self.ids(ids), use_cache=False)
            references.append(output.last_hidden_state[0].float().mean(0))
        references = torch.stack(references) if references else self.instance_queries.weight.new_empty((0, self.dim))
        return stages, references.detach()

    def validate_context(self, item, training):
        limit = int(self.base.config.max_position_embeddings)
        reserve = len(item["input_ids"]) + 2*self.capacity + self.max_steps
        length = len(item["target_ids"]) if training else self.paper_config["max_output_tokens"]
        if reserve + length > limit:
            raise ValueError("Document + instance memory + latent/bridge + output exceeds backbone context")
        if training and max([len(item["teacher_ids"])] + [len(x) for x in item["event_reference_ids"]]) > limit:
            raise ValueError("Full teacher reference exceeds backbone context; refusing silent truncation")

    def latent_path(self, cache, summary, memory, budget_condition, steps, teacher=None):
        dtype = self.base.get_input_embeddings().weight.dtype
        output, _ = self.run_backbone(inputs_embeds=memory[None].to(dtype), past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        latent = summary + self.paper_config["budget_embedding_factor"]*self.budget_embedding.weight[budget_condition-1]
        distill = summary.sum()*0
        for step in range(steps):
            routed = self.route(latent, memory)
            output, student = self.run_backbone(inputs_embeds=routed[None, None].to(dtype), past_key_values=cache,
                                                use_cache=True, capture=teacher is not None)
            cache = output.past_key_values
            latent = output.last_hidden_state[0, -1].float()
            if teacher is not None: distill = distill + process_loss(student, teacher, step, steps)
        bridge = self.bridge(latent, memory)
        output, _ = self.run_backbone(inputs_embeds=bridge[None].to(dtype), past_key_values=cache, use_cache=True)
        return output.past_key_values, output.last_hidden_state[:, -1:], distill/steps

    def forward(self, item, progress=0.0):
        self.validate_context(item, True)
        teacher, references = self.teacher_targets(item)
        cache, summary, sentences = self.encode_document(item)
        budget_logits, probabilities, expected, proposal = self.budget(summary)
        instances, anchors, exists = self.plan_instances(summary, sentences, probabilities)
        pairs = match_instances(anchors, item["gold_anchors"], instances, references, self.lm_head,
                               self.paper_config["semantic_match_factor"], self.paper_config["vocab_chunk_size"])
        instance_loss, components = instance_set_loss(anchors, exists, item["gold_anchors"], pairs)
        active = self.choose_active(anchors, exists, gold_anchors=item["gold_anchors"], pairs=pairs, progress=progress)
        target = int(item["target_steps"])
        condition = target if torch.rand((), device=self.device).item() < gold_budget_probability(progress) else proposal
        budget_loss = F.cross_entropy(budget_logits[None].float(), torch.tensor([target-1], device=self.device))
        budget_loss += self.paper_config["ordinal_factor"]*F.smooth_l1_loss(expected, expected.new_tensor(float(target)))
        cache, first, distill_loss = self.latent_path(cache, summary, instances[active], condition, target, teacher)
        labels = self.ids(item["target_ids"])
        # The last bridge token predicts the FIRST target token.
        if labels.shape[1] > 1:
            output, _ = self.run_backbone(input_ids=labels[:, :-1], past_key_values=cache, use_cache=False)
            hidden = torch.cat((first, output.last_hidden_state), dim=1)
        else: hidden = first
        generation = chunked_generation_loss(hidden, labels, self.lm_head, self.paper_config["ce_chunk_size"])
        gate = (self.paper_config["auxiliary_scale"]/generation.detach().clamp_min(1e-8)).clamp(0, 1)
        reason = self.paper_config["budget_loss_factor"]*budget_loss + self.paper_config["distill_loss_factor"]*distill_loss
        loss = generation + gate*(instance_loss + reason)
        return {"loss": loss, "generation_loss": generation.detach(), "instance_loss": instance_loss.detach(),
                "budget_loss": budget_loss.detach(), "distill_loss": distill_loss.detach(), "auxiliary_gate": gate,
                "target_steps": target, "predicted_steps": proposal, "active_instances": len(active),
                **{f"instance_{k}": v.detach() for k, v in components.items()}}

    @torch.no_grad()
    def generate(self, item, tokenizer):
        if self.training: raise ValueError("Call eval() before generation")
        self.validate_context(item, False)
        cache, summary, sentences = self.encode_document(item)
        _, probabilities, _, proposal = self.budget(summary)
        instances, anchors, exists = self.plan_instances(summary, sentences, probabilities)
        active = self.choose_active(anchors, exists)
        cache, hidden, _ = self.latent_path(cache, summary, instances[active], proposal, proposal)
        tokens, reason = [], "max_tokens"
        for _ in range(self.paper_config["max_output_tokens"]):
            logits = self.lm_head(hidden[:, -1].to(self.lm_head.weight.dtype))
            token = int(logits.argmax(-1).item())
            if token == tokenizer.eos_token_id:
                reason = "eos"; break
            tokens.append(token)
            output, _ = self.run_backbone(input_ids=self.ids([token]), past_key_values=cache, use_cache=True)
            cache, hidden = output.past_key_values, output.last_hidden_state
        return {"text": tokenizer.decode(tokens, skip_special_tokens=True), "stop_reason": reason,
                "predicted_steps": proposal, "active_instances": len(active), "generated_tokens": len(tokens)}

    def save_checkpoint(self, directory, metadata=None):
        path = Path(directory); path.mkdir(parents=True, exist_ok=True)
        trainable = {name for name, p in self.named_parameters() if p.requires_grad}
        state = {k: v.detach().cpu() for k, v in self.state_dict().items() if k in trainable}
        temporary = path/"model.pt.tmp"
        torch.save(state, temporary); temporary.replace(path/"model.pt")
        write_json(path/"paper_metadata.json", {"method_version": METHOD_VERSION, "config": self.paper_config,
                   "statistics": self.statistics, "required_weights": sorted(trainable), "metadata": metadata or {}})

    def load_checkpoint(self, directory):
        path, metadata = checkpoint_metadata(directory)
        if metadata["method_version"] != METHOD_VERSION or metadata["config"] != self.paper_config or metadata["statistics"] != self.statistics:
            raise ValueError("Checkpoint method/config/statistics mismatch; legacy weights are not paper-v1 checkpoints")
        state = torch.load(path/"model.pt", map_location="cpu", weights_only=True)
        required = {k for k, p in self.named_parameters() if p.requires_grad}
        if set(state) != required or set(metadata["required_weights"]) != required:
            raise ValueError("Checkpoint is missing paper weights or contains unexpected weights")
        result = self.load_state_dict(state, strict=False)
        if result.unexpected_keys or required.intersection(result.missing_keys): raise ValueError("Incomplete checkpoint")
        return metadata


def load_pretrained(config, statistics, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    if config["bf16"] and (str(device) != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("Paper bf16 requires supported CUDA hardware; CPU needs a labelled debug config")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)
    if not tokenizer.is_fast: raise ValueError("Exact token alignment requires a fast tokenizer")
    base = AutoModelForCausalLM.from_pretrained(config["model_name"], torch_dtype=torch.bfloat16 if config["bf16"] else torch.float32)
    adapter = LoraConfig(r=config["lora_r"], lora_alpha=config["lora_alpha"], lora_dropout=config["lora_dropout"],
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM")
    model = APLER(get_peft_model(base, adapter), config, statistics).to(device)
    for parameter in model.parameters():
        if parameter.requires_grad: parameter.data = parameter.data.float()
    return model, tokenizer
