import json
from pathlib import Path

DEBUG_DIR = Path(__file__).resolve().parents[3] / "data" / "debug"
wyniki = json.loads((DEBUG_DIR / "test_5_zawodnikow.json").read_text(encoding="utf-8"))

for w in wyniki:
    if "error" in w:
        continue
    nazwa = w.get("_nazwa", "?")
    url = w.get("_url", "")
    print(f"\n{'='*60}")
    print(f"ZAWODNIK: {nazwa}")
    print(f"URL: {url}")
    print(f"{'='*60}")
    for k, v in w.items():
        if k.startswith("_"):
            continue
        print(f"  {k:<40} {v}")