"""Formal machinery of Sec. 3 (paper numbering in brackets).

Everything here is pure and label-free; it is used by the benchmark replay
(methods.py) and the evaluation protocol.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from scipy.stats import beta as _beta
from scipy.stats import binom as _binom

# ---------------------------------------------------------------------------
# Majority answer and agreement counts
# ---------------------------------------------------------------------------

def majority(answers: Sequence[Optional[str]]) -> tuple[Optional[str], int, int]:
    """Return (majority answer, #agreements with it, #draws).

    Extraction failures (None) count as draws that do NOT agree -- the
    conservative convention (a parse failure can never raise confidence).
    Ties are broken by the earliest-appearing answer.
    """
    valid = [a for a in answers if a is not None]
    n = len(answers)
    if not valid:
        return None, 0, n
    counts = Counter(valid)
    best = max(counts.values())
    for a in valid:  # earliest among the tied
        if counts[a] == best:
            return a, best, n
    raise AssertionError  # unreachable


# ---------------------------------------------------------------------------
# Stability posterior  [Eq. 5-6]
# ---------------------------------------------------------------------------

def jeffreys_params(m: int, k: int) -> tuple[float, float]:
    return m + 0.5, k - m + 0.5


def lower_bound(m: int, k: int, alpha: float = 0.05) -> float:
    """Signal (1): lower credible bound L(alpha) of the Jeffreys posterior."""
    if k == 0:
        return 0.0
    a, b = jeffreys_params(m, k)
    return float(_beta.ppf(alpha, a, b))


def tail_mass(m: int, k: int, p0_star: float = 0.8) -> float:
    """epsilon_i = P(pbar_i < p0* | data)   [Eq. 6]."""
    a, b = jeffreys_params(m, k)
    return float(_beta.cdf(p0_star, a, b))


def tail_mass_effective(k_eff: float, p0_star: float = 0.8) -> float:
    """Tail mass for an effective unanimous depth k_eff (Prop. 3, model-based)."""
    return float(_beta.cdf(p0_star, k_eff + 0.5, 0.5))


def certification_depth(eps0: float, p0_star: float = 0.8,
                        prior: str = "jeffreys", k_max: int = 500) -> int:
    """Corollary 1: minimal unanimous depth k with eps <= eps0."""
    for k in range(1, k_max + 1):
        if prior == "jeffreys":
            eps = _beta.cdf(p0_star, k + 0.5, 0.5)
        elif prior == "uniform":
            eps = p0_star ** (k + 1)
        else:
            raise ValueError(prior)
        if eps <= eps0:
            return k
    raise ValueError("certification depth exceeds k_max")


def selection_guarantee(eps: Iterable[float]) -> dict:
    """Prop. 2: expected #sub-standard selections and the union (FWER) bound."""
    s = float(sum(eps))
    return {"expected_substandard": s, "fwer_bound": min(1.0, s)}


# ---------------------------------------------------------------------------
# Sequential triage  [Eq. 3, Lemma 3]
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SPRT:
    p0: float = 0.7
    p1: float = 0.95
    alpha: float = 0.1
    beta: float = 0.1

    @property
    def up(self) -> float:          # increment per agreement (~ +0.305)
        return math.log(self.p1 / self.p0)

    @property
    def down(self) -> float:        # increment per disagreement (~ -1.792)
        return math.log((1 - self.p1) / (1 - self.p0))

    @property
    def A(self) -> float:           # accept-H1 (certify) boundary
        return math.log((1 - self.beta) / self.alpha)

    @property
    def B(self) -> float:           # accept-H0 (eliminate) boundary
        return math.log(self.beta / (1 - self.alpha))

    def llr(self, agree: int, n: int) -> float:
        return agree * self.up + (n - agree) * self.down

    def decision(self, agree: int, n: int) -> str:
        lam = self.llr(agree, n)
        if lam >= self.A:
            return "certify"
        if lam <= self.B:
            return "eliminate"
        return "continue"

    def drift(self, p: float) -> float:
        return p * self.up + (1 - p) * self.down

    def expected_n(self) -> dict:
        """Lemma 3(b): Wald approximations of E[N] under H0 and H1."""
        e0 = (self.alpha * self.A + (1 - self.alpha) * self.B) / self.drift(self.p0)
        e1 = ((1 - self.beta) * self.A + self.beta * self.B) / self.drift(self.p1)
        return {"H0": e0, "H1": e1}


def certification_plan(sprt: "SPRT | None" = None, eps0: float = 0.05, p0_star: float = 0.8,
                       s1: int = 3) -> dict:
    """Unanimous depth a case needs to be certified: the stricter of the SPRT boundary
    (agreements until the log-LR reaches A) and Corollary 1 (eps <= eps0 at stability
    standard p0_star). With the paper's defaults: SPRT 8, Cor. 1 9 -> depth 9, s2 = 6.
    Relaxing only H0/H1 leaves Cor. 1 binding; fewer draws also need a larger eps0."""
    sprt = sprt or SPRT()
    k_sprt = math.ceil(sprt.A / sprt.up - 1e-12)
    k_cor1 = certification_depth(eps0, p0_star)
    depth = max(k_sprt, k_cor1, s1)
    return {"depth": depth, "s2": depth - s1, "sprt_depth": k_sprt, "cor1_depth": k_cor1,
            "binding": "SPRT" if k_sprt >= k_cor1 else "Cor. 1",
            "eps_at_depth": tail_mass(depth, depth, p0_star)}


def truncated_oc(p: float, s1: int = 3, s2: int = 6, sprt: SPRT | None = None) -> dict:
    """Exact operating characteristic of the two-stage (truncated) SPRT for a case of
    stability p: s1 screen draws (any disagreement eliminates), then s2 more; certify
    iff the total log-LR reaches A. Returns P(eliminate), P(certify), P(undecided)
    and E[calls]. Exact binomial arithmetic, no simulation."""
    sprt = sprt or SPRT()
    p_pass = p ** s1
    certify = sum(_binom.pmf(j, s2, p) for j in range(s2 + 1)
                  if sprt.llr(s1 + j, s1 + s2) >= sprt.A)
    return {"eliminate": 1 - p_pass, "certify": p_pass * certify,
            "undecided": p_pass * (1 - certify), "calls": s1 + p_pass * s2}


def screen_pass_probability(pbar: float, n: int = 3) -> float:
    """Lemma 3(a): the 'any disagreement eliminates' screen passes w.p. pbar^n."""
    return pbar ** n


def design_effect(m: int, rho: float) -> float:
    """Lemma 2: variance inflation 1 + (m-1) rho for m draws per paraphrase."""
    return 1 + (m - 1) * rho


# ---------------------------------------------------------------------------
# Evidence exchange  [Eq. 11-12, Prop. 3]
# ---------------------------------------------------------------------------

# Cue-follower pass probabilities q0 per probe (concept-computer: q1 = 1 - eps).
DEFAULT_Q0 = {"neg": 0.5, "relabel": 0.5, "distractor": 0.8, "heldout": 0.5}


def kappa(q0: float, q1: float = 0.95, sprt: SPRT = SPRT()) -> float:
    """Exchange rate: unanimous draws per *passed* probe."""
    return math.log(q1 / q0) / sprt.up


def probe_llr(passed: Optional[bool], q0: float, q1: float = 0.95) -> float:
    if passed is None:
        return 0.0
    return math.log(q1 / q0) if passed else math.log((1 - q1) / (1 - q0))


def evidence_llr(agree: int, n: int, probe_passes: dict[str, Optional[bool]],
                 sprt: SPRT = SPRT(), q0: dict | None = None,
                 q1: float = 0.95, shrink: float = 1.0) -> float:
    """Total log-LR = draws + probes (Prop. 3). `shrink`=0.5 is the kappa/2 haircut."""
    q0 = q0 or DEFAULT_Q0
    lam = sprt.llr(agree, n)
    for name, passed in probe_passes.items():
        lam += shrink * probe_llr(passed, q0.get(name, 0.5), q1)
    return lam


def k_effective(agree: int, probe_passes: dict[str, Optional[bool]],
                sprt: SPRT = SPRT(), q0: dict | None = None,
                q1: float = 0.95, shrink: float = 1.0) -> float:
    """Eq. 12 for unanimous draws: k_eff = m + sum of kappa over passed probes."""
    q0 = q0 or DEFAULT_Q0
    return agree + shrink * sum(kappa(q0.get(g, 0.5), q1, sprt)
                                for g, ok in probe_passes.items() if ok)


# ---------------------------------------------------------------------------
# Rationale entropy  [Eq. 9]
# ---------------------------------------------------------------------------

def miller_madow_entropy(labels: Sequence[Optional[int]]) -> float:
    """Plug-in entropy of cluster labels plus the (M-1)/(2n) bias correction.

    Draws without a cluster label (extraction/judge failure) each form their own
    singleton cluster -- conservative: missing structure never lowers entropy.
    """
    if not labels:
        return 0.0
    labs = [(-i - 1 if l is None else l) for i, l in enumerate(labels)]
    n = len(labs)
    counts = Counter(labs).values()
    h = -sum((c / n) * math.log(c / n) for c in counts)
    return h + (len(counts) - 1) / (2 * n)


# ---------------------------------------------------------------------------
# The ledger itself: an explicit, auditable account of a case's evidence
# ---------------------------------------------------------------------------

class EvidenceLedger:
    """Running account of certification evidence for one case, in nats.

    Deposits/withdrawals follow Eq. 3 (draws) and Eq. 11 (probes). The balance is
    the Prop. 3 total log-LR; `k_eff` expresses it in unanimous-draw units, and
    `certified` checks it against the SPRT boundary A. Entries are kept so every
    selection can be explained line by line.
    """

    def __init__(self, sprt: SPRT = SPRT(), q0: dict | None = None, q1: float = 0.95,
                 shrink: float = 1.0):
        self.sprt, self.q0, self.q1, self.shrink = sprt, q0 or DEFAULT_Q0, q1, shrink
        self.entries: list[tuple[str, float]] = []
        self.calls = 0

    def draw(self, agrees: bool) -> float:
        amt = self.sprt.up if agrees else self.sprt.down
        self.entries.append(("draw:agree" if agrees else "draw:disagree", amt))
        self.calls += 1
        return amt

    def probe(self, name: str, passed: Optional[bool]) -> float:
        amt = self.shrink * probe_llr(passed, self.q0.get(name, 0.5), self.q1)
        self.entries.append((f"probe:{name}:{'pass' if passed else 'fail' if passed is False else 'n/a'}", amt))
        self.calls += 1
        return amt

    @property
    def balance(self) -> float:
        return sum(a for _, a in self.entries)

    @property
    def k_eff(self) -> float:
        return self.balance / self.sprt.up

    @property
    def certified(self) -> bool:
        return self.balance >= self.sprt.A

    @property
    def closed(self) -> bool:            # eliminated: balance at/below the lower boundary
        return self.balance <= self.sprt.B

    def statement(self) -> dict:
        return {"balance_nats": self.balance, "k_eff": self.k_eff, "certified": self.certified,
                "calls": self.calls, "entries": list(self.entries)}
