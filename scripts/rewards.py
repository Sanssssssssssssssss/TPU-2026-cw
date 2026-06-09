"""Reward functions for GRPO on GSM8K.

The default ``REWARD_MODE=baseline`` preserves the original course baseline:

  1. match_format_exactly
  2. match_format_approximately
  3. check_answer
  4. check_numbers

Reward-only experiments are selected with ``REWARD_MODE``. The public
``match_format`` and ``match_numbers`` regexes are intentionally kept stable
because ``evaluate.py`` imports them as the evaluation parser.
"""

from __future__ import annotations

import math
import os
import re
from typing import Any

from data import reasoning_end, reasoning_start, solution_end, solution_start


REWARD_MODE = os.environ.get("REWARD_MODE", "baseline").strip().lower()
REWARD_DEBUG_PRINT = os.environ.get("REWARD_DEBUG_PRINT", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

REWARD_MODES = (
    "baseline",
    "no_approx",
    "light_format_oldnum",
    "numeric_primary_no_len",
    "numeric_primary_len1200",
    "numeric_primary_answer_only_len1200",
)

match_format = re.compile(
    rf"^[\s]{{0,}}"
    rf"{reasoning_start}.+?{reasoning_end}.*?"
    rf"{solution_start}(.+?){solution_end}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

match_numbers = re.compile(
    rf"{solution_start}.*?([\d\.]{{1,}})",
    flags=re.MULTILINE | re.DOTALL,
)

NUMBER_RE = re.compile(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)")
ANSWER_SECTION_RE = re.compile(
    rf"{re.escape(solution_start)}(.*?)(?:{re.escape(solution_end)}|$)",
    flags=re.MULTILINE | re.DOTALL,
)


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _first_number(text: Any) -> str | None:
    match = NUMBER_RE.search(_safe_text(text))
    if match is None:
        return None
    return match.group(0).replace(",", "")


def _float_or_none(value: Any) -> float | None:
    number = _first_number(value)
    if number is None:
        return None
    try:
        return float(number)
    except Exception:
        return None


def _answer_section(response: str) -> str | None:
    match = ANSWER_SECTION_RE.search(response)
    if match is None:
        return None
    return match.group(1)


def _extracted_number(response: str) -> str | None:
    section = _answer_section(response)
    if section is None:
        return None
    return _first_number(section)


def _numeric_flags(extracted: str | None, answer: Any) -> tuple[bool, bool]:
    got = _float_or_none(extracted)
    true = _float_or_none(answer)
    if got is None or true is None:
        return False, False
    exact = math.isclose(got, true, rel_tol=0.0, abs_tol=1e-9)
    if true == 0:
        partial = exact
    else:
        partial = 0.9 <= got / true <= 1.1
    return exact, partial


def _tag_counts(response: str) -> dict[str, int]:
    return {
        "reasoning_start": response.count(reasoning_start),
        "reasoning_end": response.count(reasoning_end),
        "solution_start": response.count(solution_start),
        "solution_end": response.count(solution_end),
    }


def _answer_tag_pair_ok(response: str) -> bool:
    return (
        response.count(solution_start) == 1
        and response.count(solution_end) == 1
        and response.find(solution_start) < response.find(solution_end)
    )


def _duplicate_or_broken_answer_tag(response: str) -> bool:
    starts = response.count(solution_start)
    ends = response.count(solution_end)
    if starts != 1 or ends != 1:
        return True
    return response.find(solution_start) > response.find(solution_end)


def _numeric_primary_score(response: str, true_answer: Any) -> float:
    if not response.strip():
        return -3.0
    extracted = _extracted_number(response)
    got = _float_or_none(extracted)
    true = _float_or_none(true_answer)
    if got is None or true is None:
        return -2.0
    if math.isclose(got, true, rel_tol=0.0, abs_tol=1e-9):
        return 5.0
    if true != 0 and 0.9 <= got / true <= 1.1:
        return 1.0
    return -1.0


def _strict_light_score(response: str) -> float:
    return 0.5 if match_format.search(response) is not None else 0.0


def _answer_tag_light_score(response: str) -> float:
    if not response.strip():
        return 0.0
    return 0.3 if _answer_tag_pair_ok(response) else -0.3


def _length_penalty_len1200(response: str) -> float:
    chars = len(response)
    if chars <= 1200:
        return 0.0
    return -min(0.75, (chars - 1200) / 800.0)


def match_format_exactly(prompts, completions, **kwargs):
    """+3 if the whole template parses, 0 otherwise."""
    return [0 if match_format.search(r) is None else 3.0 for r in completions]


def match_format_approximately(prompts, completions, **kwargs):
    """Up to +2.5 for having each of the five expected tags exactly once."""
    scores = []
    for response in completions:
        s = 0.0
        s += 0.5 if response.count(reasoning_start) == 1 else -0.5
        s += 0.5 if response.find(reasoning_start) == 0 else -0.5
        s += 0.5 if response.count(reasoning_end) == 1 else -0.5
        s += 0.5 if response.count(solution_start) == 1 else -0.5
        s += 0.5 if response.count(solution_end) == 1 else -0.5
        scores.append(s)
    return scores


def check_answer(prompts, completions, answer, **kwargs):
    """Reward correctness of the bracketed answer with partial credit."""
    extracted = [
        guess.group(1) if r is not None and (guess := match_format.search(r)) is not None else None
        for r in completions
    ]
    assert len(extracted) == len(answer)

    scores = []
    for guess, true in zip(extracted, answer):
        if guess is None:
            scores.append(0)
            continue
        if guess == true:
            scores.append(3.0)
        elif guess.strip() == true.strip():
            scores.append(1.5)
        else:
            try:
                ratio = float(guess) / float(true)
                if 0.9 <= ratio <= 1.1:
                    scores.append(0.5)
                elif 0.8 <= ratio <= 1.2:
                    scores.append(0.25)
                else:
                    scores.append(-1.0)
            except Exception:
                scores.append(-0.5)
    return scores


def check_numbers(prompts, completions, answer, **kwargs):
    """Fallback: extract any number after <answer> and compare numerically."""
    extracted = [
        guess.group(1) if (guess := match_numbers.search(r)) is not None else None
        for r in completions
    ]

    if REWARD_DEBUG_PRINT:
        question = kwargs.get("question", [""])
        print("START ============================")
        print(f"Question:\t{question[0] if question else ''}")
        print(f"Answer:\t{answer[0] if answer else ''}")
        print(f"Response:\t{completions[0] if completions else ''}")
        print(f"Extracted:\t{extracted[0] if extracted else ''}")
        print("END ==============================")

    scores = []
    for guess, true in zip(extracted, answer):
        if guess is None:
            scores.append(0)
            continue
        try:
            scores.append(1.5 if float(guess.strip()) == float(true.strip()) else 0.0)
        except Exception:
            scores.append(0)
    return scores


def match_format_strict_light(prompts, completions, **kwargs):
    """Small shaping reward for the full course response template."""
    return [_strict_light_score(response) for response in completions]


def match_answer_tag_light(prompts, completions, **kwargs):
    """Small answer-tag reward; duplicate, missing, or reversed tags are penalized."""
    return [_answer_tag_light_score(response) for response in completions]


def numeric_primary(prompts, completions, answer, **kwargs):
    """Correctness-dominant numeric reward used by the reward sweep."""
    return [_numeric_primary_score(response, true) for response, true in zip(completions, answer)]


def length_penalty_1200(prompts, completions, **kwargs):
    """Mild penalty for responses longer than 1200 characters."""
    return [_length_penalty_len1200(response) for response in completions]


def _baseline_components(response: str, answer: Any) -> dict[str, Any]:
    format_match = match_format.search(response)
    number_match = match_numbers.search(response)
    formatted_answer = format_match.group(1) if format_match is not None else None
    extracted_number = number_match.group(1) if number_match is not None else None

    exact_format = 0.0 if format_match is None else 3.0

    approx_format = 0.0
    approx_format += 0.5 if response.count(reasoning_start) == 1 else -0.5
    approx_format += 0.5 if response.find(reasoning_start) == 0 else -0.5
    approx_format += 0.5 if response.count(reasoning_end) == 1 else -0.5
    approx_format += 0.5 if response.count(solution_start) == 1 else -0.5
    approx_format += 0.5 if response.count(solution_end) == 1 else -0.5

    answer_score = 0.0
    true = _safe_text(answer)
    if formatted_answer is not None and true:
        if formatted_answer == true:
            answer_score = 3.0
        elif formatted_answer.strip() == true.strip():
            answer_score = 1.5
        else:
            try:
                ratio = float(formatted_answer) / float(true)
                if 0.9 <= ratio <= 1.1:
                    answer_score = 0.5
                elif 0.8 <= ratio <= 1.2:
                    answer_score = 0.25
                else:
                    answer_score = -1.0
            except Exception:
                answer_score = -0.5

    number_score = 0.0
    if extracted_number is not None and true:
        try:
            number_score = 1.5 if float(extracted_number.strip()) == float(true.strip()) else 0.0
        except Exception:
            number_score = 0.0

    return {
        "match_format_exactly": exact_format,
        "match_format_approximately": approx_format,
        "check_answer": answer_score,
        "check_numbers": number_score,
        "formatted_answer": formatted_answer,
        "baseline_extracted_number": extracted_number,
        "format_ok": format_match is not None,
    }


def reward_diagnostics_for_observability(
    completions: list[str],
    answers: list[Any],
    reward_mode: str | None = None,
) -> list[dict[str, Any]]:
    """Return per-completion reward components and parse flags for logging."""
    mode = reward_mode or REWARD_MODE
    rows = []
    for completion_raw, answer in zip(completions, answers):
        completion = _safe_text(completion_raw)
        baseline = _baseline_components(completion, answer)
        extracted_number = _extracted_number(completion)
        numeric_exact, numeric_partial = _numeric_flags(extracted_number, answer)
        tag_counts = _tag_counts(completion)
        answer_numbers = NUMBER_RE.findall(_answer_section(completion) or "")

        numeric_primary_score = _numeric_primary_score(completion, answer)
        format_strict_light_score = _strict_light_score(completion)
        answer_tag_light_score = _answer_tag_light_score(completion)
        length_penalty_score = _length_penalty_len1200(completion)
        format_light_total = format_strict_light_score + answer_tag_light_score

        component_values = {
            "match_format_exactly": baseline["match_format_exactly"],
            "match_format_approximately": baseline["match_format_approximately"],
            "check_answer": baseline["check_answer"],
            "check_numbers": baseline["check_numbers"],
            "format_strict_light": format_strict_light_score,
            "answer_tag_light": answer_tag_light_score,
            "numeric_primary": numeric_primary_score,
            "length_penalty_1200": length_penalty_score,
        }
        mode_components = reward_components_for_mode(mode)
        mode_reward = float(sum(component_values[name] for name in mode_components))

        rows.append(
            {
                **baseline,
                "reward_mode": mode,
                "reward_components_for_mode": mode_components,
                "reward_total_recomputed": mode_reward,
                "format_strict_light": format_strict_light_score,
                "answer_tag_light": answer_tag_light_score,
                "numeric_primary": numeric_primary_score,
                "length_penalty_1200": length_penalty_score,
                "format_light_total": format_light_total,
                "extracted_number": extracted_number,
                "numeric_exact": numeric_exact,
                "numeric_partial": numeric_partial,
                "answer_tag_pair_ok": _answer_tag_pair_ok(completion),
                "duplicate_or_broken_answer_tag": _duplicate_or_broken_answer_tag(completion),
                "overlong_1200": len(completion) > 1200,
                "overlong_1600": len(completion) > 1600,
                "answer_multi_number": len(answer_numbers) > 1,
                "tag_counts": tag_counts,
            }
        )
    return rows


def reward_components_for_mode(mode: str) -> list[str]:
    mode = mode.strip().lower()
    if mode == "baseline":
        return ["match_format_exactly", "match_format_approximately", "check_answer", "check_numbers"]
    if mode == "no_approx":
        return ["match_format_exactly", "check_answer", "check_numbers"]
    if mode == "light_format_oldnum":
        return ["format_strict_light", "answer_tag_light", "check_answer", "check_numbers"]
    if mode == "numeric_primary_no_len":
        return ["numeric_primary", "format_strict_light", "answer_tag_light"]
    if mode == "numeric_primary_len1200":
        return ["numeric_primary", "format_strict_light", "answer_tag_light", "length_penalty_1200"]
    if mode == "numeric_primary_answer_only_len1200":
        return ["numeric_primary", "answer_tag_light", "length_penalty_1200"]
    raise ValueError(f"Unknown REWARD_MODE '{mode}'. Valid modes: {', '.join(REWARD_MODES)}")


def reward_functions_for_mode(mode: str):
    mode = mode.strip().lower()
    if mode == "baseline":
        return [match_format_exactly, match_format_approximately, check_answer, check_numbers]
    if mode == "no_approx":
        return [match_format_exactly, check_answer, check_numbers]
    if mode == "light_format_oldnum":
        return [match_format_strict_light, match_answer_tag_light, check_answer, check_numbers]
    if mode == "numeric_primary_no_len":
        return [numeric_primary, match_format_strict_light, match_answer_tag_light]
    if mode == "numeric_primary_len1200":
        return [numeric_primary, match_format_strict_light, match_answer_tag_light, length_penalty_1200]
    if mode == "numeric_primary_answer_only_len1200":
        return [numeric_primary, match_answer_tag_light, length_penalty_1200]
    raise ValueError(f"Unknown REWARD_MODE '{mode}'. Valid modes: {', '.join(REWARD_MODES)}")


REWARD_FNS = reward_functions_for_mode(REWARD_MODE)
