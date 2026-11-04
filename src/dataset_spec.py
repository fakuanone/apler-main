"""Versioned schema, split identity, and paper configuration contracts."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHOD_VERSION = "apler-paper-v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_schema(path=None):
    data = read_json(path or ROOT / "configs/schemas/emergency_ee.json")
    return data["events"]


def validate_rows(rows, schema):
    seen = set()
    for row in rows:
        doc_id = row.get("doc_id")
        if not doc_id or doc_id in seen:
            raise ValueError(f"Missing or duplicate doc_id: {doc_id}")
        seen.add(doc_id)
        if not isinstance(row.get("sentences"), list) or not row["sentences"] or any(not isinstance(s, str) for s in row["sentences"]):
            raise ValueError(f"{doc_id}: sentences must be nonempty")
        if not isinstance(row.get("events"), list):
            raise ValueError(f"{doc_id}: events must be a list, including for zero-event rows")
        for event in row.get("events", []):
            kind = event["event_type"]
            if kind not in schema:
                raise ValueError(f"{doc_id}: unsupported type {kind}")
            unknown = set(event["arguments"]) - set(schema[kind]["roles"])
            if unknown:
                raise ValueError(f"{doc_id}: unsupported roles {unknown}")
            if not isinstance(event.get("trigger"), str) or not event["trigger"].strip():
                raise ValueError(f"{doc_id}: missing trigger text")
            if any(not isinstance(v, list) or any(not isinstance(x, str) or not x.strip() for x in v) for v in event["arguments"].values()):
                raise ValueError(f"{doc_id}: arguments must be lists of source text values")
            if not event.get("evidence") or any(not re.fullmatch(r"S\d+", str(s)) or int(s[1:]) >= len(row["sentences"]) for s in event["evidence"]):
                raise ValueError(f"{doc_id}: event requires valid evidence sentence IDs")


def load_partitions(manifest_path, schema):
    path = Path(manifest_path).resolve()
    manifest = read_json(path)
    if manifest.get("method_version") != METHOD_VERSION:
        raise ValueError("Expected a paper-v1 split manifest, not a legacy selection manifest")
    source = (path.parent / manifest["source"].replace("\\", "/")).resolve()
    if sha256(source) != manifest["source_sha256"]:
        raise ValueError("Dataset source hash differs from split manifest")
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    validate_rows(rows, schema)
    lookup = {r["doc_id"]: r for r in rows}
    ids = manifest["splits"]
    if set(ids) != {"train", "dev", "test"} or any(not ids[k] for k in ids):
        raise ValueError("Manifest needs nonempty train/dev/test lists")
    flat = [i for values in ids.values() for i in values]
    if len(flat) != len(set(flat)) or set(flat) != set(lookup):
        raise ValueError("Split IDs must form a disjoint, complete partition of source")
    owners = {}
    for split, values in ids.items():
        for doc_id in values:
            row = lookup[doc_id]
            links = [f"source:{v}" for v in row.get("source_doc_ids", [])]
            links.append("text:" + hashlib.sha256("\n".join(row["sentences"]).encode("utf-8")).hexdigest())
            if row.get("cluster_id") is not None:
                links.append(f"cluster:{row['cluster_id']}")
            for key in links:
                if key in owners and owners[key] != split:
                    raise ValueError(f"Source/cluster overlap across splits: {key}")
                owners[key] = split
    return {k: [lookup[i] for i in values] for k, values in ids.items()}, manifest


def load_config(path):
    config = read_json(path)
    if config.get("method_version") != METHOD_VERSION:
        raise ValueError("Unsupported method version")
    if config["min_steps"] != 1 or config["max_steps"] != 9:
        raise ValueError("Paper configuration requires latent budgets 1..9")
    return config


def code_identity():
    paths = [ROOT/"train.py", ROOT/"test.py"] + sorted((ROOT/"src").glob("*.py"))
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in paths}
