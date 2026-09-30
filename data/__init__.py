"""All data preprocessing: dataset download/checks, verification-format items, dev/eval split.

  loaders.py  HF datasets -> binary verification items (TriviaQA de-duplicated)
  splits.py   dev slice / eval items / batches
  prepare.py  download + schema checks + item manifests (python -m data.prepare)
"""
from .loaders import load_items, synthetic_items
from .splits import Split, load_split, make_split

__all__ = ["load_items", "synthetic_items", "Split", "load_split", "make_split"]
