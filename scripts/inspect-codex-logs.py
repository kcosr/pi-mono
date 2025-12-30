#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path


def default_log_path() -> str:
    return os.environ.get("PI_LOG_REQUESTS_PATH") or os.path.expanduser("~/.pi/agent/pi-requests.jsonl")


def load_entries(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")
    entries = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"warning: failed to parse JSONL line {line_no}: {exc}", file=sys.stderr)
    return entries


def first_input_text(entry: dict) -> str:
    params = entry.get("params", {}) if isinstance(entry, dict) else {}
    input_items = params.get("input") or []
    if not input_items:
        return ""
    first = input_items[0]
    if isinstance(first, str):
        return first
    if not isinstance(first, dict):
        return str(first)
    content = first.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content:
        part = content[0]
        if isinstance(part, dict):
            return part.get("text") or part.get("output_text") or ""
    return ""


def filter_entries(entries, provider: str | None, model: str | None):
    if not provider and not model:
        return entries
    filtered = []
    for entry in entries:
        if provider and entry.get("provider") != provider:
            continue
        if model and entry.get("model") != model:
            continue
        filtered.append(entry)
    return filtered


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect pi request logs (JSONL) written by --log-requests or PI_LOG_REQUESTS_PATH."
    )
    parser.add_argument("--path", default=default_log_path(), help="Log file path (default: env or ~/.pi/agent)")
    parser.add_argument("--tail", type=int, default=3, help="Number of entries to summarize (default: 3)")
    parser.add_argument("--dump", action="store_true", help="Dump full JSON for the last entry")
    parser.add_argument("--dump-index", type=int, help="Dump full JSON for entry index (0-based)")
    parser.add_argument("--filter-provider", help="Filter entries by provider")
    parser.add_argument("--filter-model", help="Filter entries by model")

    args = parser.parse_args()
    path = Path(args.path)

    try:
        entries = load_entries(path)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    entries = filter_entries(entries, args.filter_provider, args.filter_model)
    if not entries:
        print("No entries found.")
        return 0

    if args.dump_index is not None:
        idx = args.dump_index
        if idx < 0 or idx >= len(entries):
            print(f"Index out of range: {idx}", file=sys.stderr)
            return 1
        print(json.dumps(entries[idx], indent=2, sort_keys=True))
        return 0

    if args.dump:
        print(json.dumps(entries[-1], indent=2, sort_keys=True))
        return 0

    tail = max(1, args.tail)
    for entry in entries[-tail:]:
        provider = entry.get("provider", "")
        model = entry.get("model", "")
        text = first_input_text(entry).replace("\n", " ")
        text = text[:60]
        print(f"{provider} {model} {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
