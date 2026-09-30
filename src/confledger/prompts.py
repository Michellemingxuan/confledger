"""Prompt templates. Every template is freezable on the dev slice (Sec. 4.1).

Each builder returns (prompt_text, variant_tag). The variant tag travels in the
request metadata so the simulator (and logs) know which transform was applied.
"""
from __future__ import annotations

import hashlib

ANSWER_FORMAT_YN = (
    "Respond in exactly this format:\n"
    "Answer: <Yes or No>\n"
    "Explanation: <one or two sentences giving your reason>"
)

ANSWER_FORMAT_AB = (
    "Respond with option A if the answer is No, and option B if the answer is Yes.\n"
    "Respond in exactly this format:\n"
    "Answer: <A or B>\n"
    "Explanation: <one or two sentences giving your reason>"
)

# Unrelated true statements for the distractor-insertion probe (sigma_g = id).
DISTRACTORS = [
    "The Pacific Ocean is the largest ocean on Earth.",
    "Honey stored in sealed containers can remain edible for a very long time.",
    "A standard chessboard has sixty-four squares.",
    "Mount Kilimanjaro is located in Tanzania.",
    "Octopuses have three hearts.",
    "The Great Wall of China was built over many centuries.",
]


def _pick(seq, key: str):
    h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
    return seq[h % len(seq)]


def answer_prompt(question: str, with_confidence: bool = False) -> str:
    """The draw prompt. with_confidence adds the verbal-confidence line to every draw
    (collect.verbal_per_draw), replacing the separate verbal call."""
    fmt = ANSWER_FORMAT_YN
    if with_confidence:
        fmt += "\nConfidence: <an integer from 0 to 100, the probability your answer is correct>"
    return f"{question}\n\n{fmt}"


# --- label-action probes (Sec. 3.4, signal 2) -------------------------------

def negation_prompt(question: str) -> str:
    """Structural negation: the correct answer flips (sigma_g(y) = 1 - y)."""
    return (f"Consider the following question:\n\"{question}\"\n\n"
            f"Is the correct answer to that question \"No\"?\n\n{ANSWER_FORMAT_YN}")


def relabel_prompt(question: str) -> str:
    """Option relabeling / reordering: answer mapped back, sigma_g = id after mapping."""
    return f"{question}\n\n{ANSWER_FORMAT_AB}"


def distractor_prompt(question: str, key: str) -> str:
    """Irrelevant context insertion: sigma_g = id."""
    fact = _pick(DISTRACTORS, key)
    return f"Background (may be irrelevant): {fact}\n\n{question}\n\n{ANSWER_FORMAT_YN}"


def heldout_prompt(question: str) -> str:
    """Held-out family (never used in selection): conversational re-framing.

    Label-preserving by construction. NOTE: the draft's 'entity renaming' family
    is not label-preserving for factual recall (renaming changes the true answer),
    so the held-out family is a framing shift instead.
    """
    return (f"A colleague asked me this and I want to double-check before replying:\n"
            f"{question}\n\n{ANSWER_FORMAT_YN}")


# --- generation-side prompts -------------------------------------------------

def paraphrase_gen_prompt(question: str, n: int) -> str:
    return (
        f"Rewrite the following yes/no question in {n} different ways. Every rewrite "
        f"must preserve the exact meaning and the correct answer; vary wording and "
        f"sentence structure only. Do not answer the question.\n\n"
        f"Question: {question}\n\n"
        f"Return ONLY a JSON list of {n} strings, with no other text."
    )


def distractor_answer_prompt(question: str, gold_answer: str) -> str:
    return (
        f"Question: {question}\nThe correct answer is: {gold_answer}\n\n"
        f"Give ONE plausible but incorrect answer of the same type (e.g. another "
        f"person, place, or date). Respond with only the answer, nothing else."
    )


def challenge_prompt(question: str, answer: str) -> str:
    ans = "Yes" if answer == "yes" else "No"
    return (
        f"{question}\n\nYou previously answered: {ans}.\n"
        f"A reviewer disagrees and argues that this answer is wrong and that you "
        f"overlooked something important. Reconsider carefully, then give your final answer.\n\n"
        f"{ANSWER_FORMAT_YN}"
    )


def verbal_prompt(question: str) -> str:
    return (
        f"{question}\n\nRespond in exactly this format:\n"
        f"Answer: <Yes or No>\n"
        f"Confidence: <an integer from 0 to 100, the probability your answer is correct>"
    )


def judge_prompt(question: str, explanations: list[str]) -> str:
    listing = "\n".join(f"[{i}] {e}" for i, e in enumerate(explanations))
    return (
        f"Below are {len(explanations)} explanations given for answers to the question:\n"
        f"\"{question}\"\n\n{listing}\n\n"
        f"Group the explanations by the underlying REASON they give (not their wording or "
        f"final answer). Two explanations belong to the same group if they rely on the "
        f"same argument.\n"
        f"Return ONLY a JSON list of {len(explanations)} integers: the group id of each "
        f"explanation in order, with group ids starting at 0."
    )
