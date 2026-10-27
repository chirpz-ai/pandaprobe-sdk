"""Pin the API contracts used by this SDK from an exported backend OpenAPI JSON.

Run: python scripts/update-platform-contract.py /path/to/openapi.json
This development utility neither imports backend code nor contacts a server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

OPERATIONS = {
    "/traces": "get",
    "/traces/{trace_id}": "get",
    "/evaluations/trace-metrics": "get",
    "/evaluations/trace-runs/batch": "post",
    "/evaluations/trace-runs/{run_id}": "get",
    "/evaluations/trace-runs/{run_id}/scores": "get",
}


def references(value):
    if isinstance(value, dict):
        if "$ref" in value:
            yield value["$ref"].rsplit("/", 1)[-1]
        for child in value.values():
            yield from references(child)
    elif isinstance(value, list):
        for child in value:
            yield from references(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("openapi", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[1] / "test/fixtures/platform_contract.json"
    )
    args = parser.parse_args()
    raw = args.openapi.read_bytes()
    document = json.loads(raw)
    operations = {}
    for path, method in OPERATIONS.items():
        operation = document["paths"][path][method]
        operations[f"{method.upper()} {path}"] = {
            key: operation[key] for key in ("parameters", "requestBody", "responses") if key in operation
        }
    schemas = {}
    pending = set(references(operations))
    while pending:
        name = pending.pop()
        schemas[name] = document["components"]["schemas"][name]
        pending.update(set(references(schemas[name])) - schemas.keys())
    creation = operations["POST /evaluations/trace-runs/batch"]
    fixture = {
        "source": "Exported current backend app.openapi(); refresh with scripts/update-platform-contract.py",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "paths": {path: [method] for path, method in OPERATIONS.items()},
        "operations": operations,
        "schemas": schemas,
        "creation_headers": [
            parameter["name"] for parameter in creation.get("parameters", []) if parameter["in"] == "header"
        ],
    }
    args.output.write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
