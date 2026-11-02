"""Baselines (Sec. 4.4), one module per source. Each is a replay policy
(CaseData, ..., seed) -> CaseScore with its own call count, so run_protocol matches it to
CEL's budget. Fitted baselines (CISC's temperature, Pedapati's regression) are frozen on
D_fit (the dev slice) before any eval case is scored.

  self_consistency.py  Wang et al. 2023: majority vote, agreement share; + verbal tie-break
  xiong.py             Xiong et al. 2024 (ICLR): sampling + verbal confidence, Avg-Conf
  cisc.py              Taubenfeld et al. 2025: confidence-weighted self-consistency (CISC)
  pedapati.py          Pedapati et al. 2024: perturbation features + logistic regression
  others.py            verbalized, semantic entropy, BSDetector-style, random

All replay the existing collection (no new prompts). Variants that need new prompts are
not implemented: Xiong's misleading hints and Top-K/Pair-Rank, CISC's P(True), and
Pedapati's sentence permutation, entity-frequency, stop-word and NLI split-response features.
"""
from .cisc import cisc, fit_cisc
from .others import bsdetector_draws, bsdetector_like, random_selection, semantic_entropy, verbalized
from .pedapati import fit_pedapati, pedapati
from .self_consistency import self_consistency, self_consistency_verbal
from .xiong import xiong

__all__ = ["self_consistency", "self_consistency_verbal", "xiong", "cisc", "fit_cisc", "pedapati",
           "fit_pedapati", "verbalized", "semantic_entropy", "bsdetector_like", "bsdetector_draws", "random_selection"]
