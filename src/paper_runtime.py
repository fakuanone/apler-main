"""Free-running validation shared by training and standalone evaluation."""
from src.paper_data import tokenize_example
from src.paper_metrics import parse_prediction, score_document, aggregate


def evaluate(model, tokenizer, rows, schema, config, statistics):
    was_training = model.training
    model.eval()
    records = []
    try:
        for row in rows:
            # Explicitly strip ALL gold metadata before entering the prediction path.
            inference_input = {"doc_id": row["doc_id"], "sentences": row["sentences"]}
            item = tokenize_example(inference_input, tokenizer, schema, config, statistics)
            prediction = model.generate(item, tokenizer)
            payload = parse_prediction(prediction["text"])
            scores = score_document(row, payload, mode=config.get("evaluation_mode", "paper_span"))
            records.append({"doc_id": row["doc_id"], **scores, "prediction": prediction})
    finally:
        model.train(was_training)
    return {"evaluation_mode": config.get("evaluation_mode", "paper_span"),
            "summary": aggregate(records), "documents": records}


def learning_rate_factor(step, total_steps, warmup_steps, initial, floor):
    import math
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return floor/initial + (1-floor/initial)*.5*(1+math.cos(math.pi*progress))


class DevelopmentSelection:
    def __init__(self, patience=2):
        self.patience, self.best, self.bad_epochs = patience, -1.0, 0

    def update(self, arg_c):
        improved = arg_c > self.best
        if improved: self.best, self.bad_epochs = arg_c, 0
        else: self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience
