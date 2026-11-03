"""Check a config before spending: transport, datasets, and one short uncached call to each
model. Costs one call per model. Prints no keys or tokens.

    python experiments/scripts/preflight.py -c configs/full_gpt41.yaml
"""
import os

from confledger.utility.config import arg_parser, check_config, from_args, model_spec
from confledger.utility.llm import Request, make_backend
from data.loaders import SOURCES_DIR, load_source, verify_sources

cfg = from_args(arg_parser(__doc__).parse_args())
check_config(cfg)
print(f"LLM_BACKEND = {os.environ.get('LLM_BACKEND') or '(unset: direct API)'}")
failed = []


def check(label, fn):
    try:
        print(f"ok    {label}: {fn()}")
    except Exception as e:
        failed.append(label)
        print(f"FAIL  {label}: {type(e).__name__}: {str(e)[:200]}")


def dataset(ds):
    local = (SOURCES_DIR / f"{ds}.jsonl.gz").exists()
    if local and (SOURCES_DIR / "manifest.json").exists():
        verify_sources([ds])                                  # sha256 and row count against manifest.json
    return f"{len(load_source(ds)):,} rows from " + (f"local export {SOURCES_DIR}" if local else "Hugging Face")


def reply(spec):
    backend = make_backend(spec)
    text = backend.generate(Request("Reply with the single word OK.", 0.0, "preflight", 0, 20))
    if not text.strip():
        raise ValueError("empty reply")
    return f"{backend.name} replied {text.strip()[:30]!r}"


for ds in cfg["_datasets"]:
    check(f"dataset {ds}", lambda ds=ds: dataset(ds))
for role, spec in [(f"target {mk}", model_spec(cfg, mk)) for mk in cfg["_models"]] + \
                  [("generator", cfg["generator"]), ("judge", cfg["judge"])]:
    check(role, lambda spec=spec: reply(spec))
raise SystemExit(f"preflight failed: {', '.join(failed)}" if failed else "preflight passed")
