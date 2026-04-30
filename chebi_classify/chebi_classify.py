#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from collections import deque

import requests

DEFAULT_OBO = "chebi.obo"
DEFAULT_OUT = "chebi_classifications.json"

INCHIKEY_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")


def infer_input_type(value: str) -> str:
    if value.startswith("InChI="):
        return "inchi"
    if INCHIKEY_RE.match(value.strip()):
        return "inchikey"
    return "smiles"


def fetch_chebi_match(value: str, value_type: str) -> dict | None:
    url = "https://www.ebi.ac.uk/chebi/backend/api/public/advanced_search/"
    body = {
        "text_search_specification": {
            "and_specification": [
                {
                    "text": value,
                    "category": value_type,
                }
            ]
        }
    }

    resp = requests.post(url, json=body, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"ChEBI API error: HTTP {resp.status_code} - {resp.text[:200]}")

    data = resp.json()
    results = data.get("results", [])
    if not results:
        return None

    # Pick highest score (results are generally sorted by score already).
    best = results[0].get("_source", {})
    return {
        "chebi_accession": best.get("chebi_accession"),
        "name": best.get("name"),
        "inchi": best.get("inchi"),
        "inchikey": best.get("inchikey"),
        "smiles": best.get("smiles"),
    }


def parse_obo(obo_path: str):
    parents = {}
    names = {}
    current_id = None

    with open(obo_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line == "[Term]":
                current_id = None
                continue
            if not line:
                current_id = None
                continue
            if line.startswith("id: CHEBI:"):
                current_id = line.split("id: ", 1)[1].strip()
                parents.setdefault(current_id, [])
                continue
            if current_id is None:
                continue
            if line.startswith("name: "):
                names[current_id] = line.split("name: ", 1)[1].strip()
                continue
            if line.startswith("is_a: CHEBI:"):
                parent = line.split("is_a: ", 1)[1].split(" ! ", 1)[0].strip()
                parents[current_id].append(parent)
                continue

    return parents, names


def collect_ancestors(start_id: str, parents: dict):
    seen = set()
    queue = deque(parents.get(start_id, []))
    while queue:
        current = queue.popleft()
        if current in seen:
            continue
        seen.add(current)
        for parent in parents.get(current, []):
            if parent not in seen:
                queue.append(parent)
    return seen


def write_output(path: str, payload: dict):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def main():
    parser = argparse.ArgumentParser(
        description="Return ChEBI classifications for a compound using chebi.obo."
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "SMILES, InChI, or InChIKey. If this points to a file, each non-empty "
            "line will be treated as an input value."
        ),
    )
    parser.add_argument(
        "--type",
        choices=["smiles", "inchi", "inchikey"],
        help="Input type; if omitted, inferred.",
    )
    parser.add_argument("--obo", default=DEFAULT_OBO, help="Path to chebi.obo")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output JSON file")

    args = parser.parse_args()

    if not os.path.exists(args.obo):
        payload = {
            "error": (
                "chebi.obo not found. Download 'chebi full' from "
                "https://www.ebi.ac.uk/chebi/downloads"
            )
        }
        write_output(args.out, payload)
        print(payload["error"])
        sys.exit(1)

    if os.path.isfile(args.input):
        with open(args.input, "r", encoding="utf-8") as handle:
            values = [line.strip() for line in handle if line.strip()]
    else:
        values = [args.input]

    parents, names = parse_obo(args.obo)
    results = []

    for value in values:
        value_type = args.type or infer_input_type(value)
        try:
            match = fetch_chebi_match(value, value_type)
        except Exception as exc:
            results.append(
                {
                    "input": value,
                    "input_type": value_type,
                    "error": str(exc),
                }
            )
            continue

        if not match or not match.get("chebi_accession"):
            results.append(
                {
                    "input": value,
                    "input_type": value_type,
                    "error": f"No ChEBI match found for {value_type}: {value}",
                }
            )
            continue

        chebi_id = match["chebi_accession"]
        ancestors = collect_ancestors(chebi_id, parents)
        classifications = [
            {"id": cid, "name": names.get(cid)} for cid in sorted(ancestors)
        ]

        results.append(
            {
                "input": value,
                "input_type": value_type,
                "match": {
                    "chebi_id": chebi_id,
                    "name": match.get("name"),
                    "smiles": match.get("smiles"),
                    "inchi": match.get("inchi"),
                    "inchikey": match.get("inchikey"),
                },
                "classifications": classifications,
            }
        )

    payload = {"results": results}
    write_output(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
