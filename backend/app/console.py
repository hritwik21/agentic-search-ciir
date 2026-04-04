from __future__ import annotations

import json


def demo_print(stage: str, payload: object | None = None) -> None:
    print(f"\n=== {stage} ===", flush=True)
    if payload is None:
        return
    if isinstance(payload, str):
        print(payload, flush=True)
        return
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
