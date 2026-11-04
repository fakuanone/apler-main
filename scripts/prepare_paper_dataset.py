"""Audit the collection or bind EXPLICIT study partitions; never invent splits."""
import argparse
import json
import os
from pathlib import Path
import sys
import random
import hashlib
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset_spec import load_schema, validate_rows, write_json, sha256, METHOD_VERSION, load_partitions
from src.latent_budget import training_statistics
from src.span_grounding import audit_gold


def source_group_split(rows, seed=42):
    """Group shared known sources/clusters/exact duplicate documents, then allocate."""
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    owners = {}
    for i, row in enumerate(rows):
        keys = ["source:"+str(s) for s in row.get("source_doc_ids", [])]
        keys.append("text:"+hashlib.sha256("\n".join(row["sentences"]).encode()).hexdigest())
        if row.get("cluster_id") is not None: keys.append("cluster:"+str(row["cluster_id"]))
        for key in keys:
            if key in owners: parent[find(i)] = find(owners[key])
            else: owners[key] = i
    groups = {}
    for i, row in enumerate(rows): groups.setdefault(find(i), []).append(row["doc_id"])
    groups = list(groups.values()); random.Random(seed).shuffle(groups); groups.sort(key=len, reverse=True)
    if len(groups) < 3: raise ValueError("Fewer than three independent source groups; cannot split without leakage")
    splits = {"train": [], "dev": [], "test": []}
    targets = {"train": .8*len(rows), "dev": .1*len(rows), "test": .1*len(rows)}
    for index, group in enumerate(groups):
        empty = [s for s in splits if not splits[s]]
        if len(groups)-index <= len(empty): split = empty[0]
        else: split = max(splits, key=lambda s: targets[s]-len(splits[s]))
        splits[split].extend(group)
    return {k: sorted(v) for k, v in splits.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--splits", help='JSON with train/dev/test doc_id lists; optional for audit-only')
    parser.add_argument("--create-splits", action="store_true", help="Create a NEW deterministic 80/10/10 source-group partition")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--provenance", help="Required with --splits: original study manifest or explicitly new partition")
    parser.add_argument("--schema")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = Path(args.source).resolve(), Path(args.output).resolve()
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    schema = load_schema(args.schema); validate_rows(rows, schema)
    output.mkdir(parents=True, exist_ok=True)
    audit = audit_gold(rows)
    audit.update(zero_event_documents=sum(not r["events"] for r in rows),
                 documents_above_9=sum(len(r["events"]) > 9 for r in rows),
                 source_sha256=sha256(source), schema=schema)
    write_json(output/"span_audit.json", audit)
    if args.splits or args.create_splits:
        if args.splits and args.create_splits: parser.error("Choose --splits or --create-splits")
        if args.splits and not args.provenance: parser.error("--provenance is required with --splits")
        splits = source_group_split(rows, args.seed) if args.create_splits else json.loads(Path(args.splits).read_text(encoding="utf-8-sig"))
        manifest = {"method_version": METHOD_VERSION, "source": Path(os.path.relpath(source, output)).as_posix(),
                    "source_sha256": sha256(source), "provenance": args.provenance or "new source-group partition of supplied data; not the original paper split",
                    "seed": args.seed, "splits": splits}
        candidate = output/"manifest.json"
        if candidate.exists(): raise FileExistsError("Refusing to overwrite an existing split manifest")
        pending = output/"manifest.pending.json"
        write_json(pending, manifest)
        partitions, _ = load_partitions(pending, schema)
        pending.replace(candidate)
        write_json(output/"training_statistics.json", training_statistics(partitions["train"]))
        print("Partition sizes:", {k:len(v) for k,v in partitions.items()})
    print(json.dumps({k: audit[k] for k in ("documents", "valid_documents", "zero_event_documents", "documents_above_9")}, ensure_ascii=False))
    print(f"Span audit: {output / 'span_audit.json'}")
    if not args.splits and not args.create_splits: print("Audit only: no train/dev/test partition was invented.")


if __name__ == "__main__": main()
