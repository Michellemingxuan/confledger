"""Export the three source splits to data/sources/*.jsonl.gz, for machines without Hugging Face
access (decisions.md #38). Run on a machine that has them (the Hugging Face cache is enough);
commit data/sources, or copy it to the other machine.

    python -m data.export_sources              # export all three and write manifest.json
    python -m data.export_sources --verify     # check the files against manifest.json
"""
import argparse
import json

from data.loaders import SOURCES_DIR, export_sources, verify_sources

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--verify", action="store_true", help="only check the existing files against manifest.json")
a = ap.parse_args()
print(json.dumps(verify_sources() if a.verify else export_sources(), indent=1))
print(f"{'verified' if a.verify else 'wrote'} {SOURCES_DIR}")
