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
    "numeric_dense_lastnum",
    "numeric_dense_single_answer",
    "numeric_dense_single_answer_short",
    "closed_answer_minimal",
    "numeric_guarded",
    "numeric_guarded_fallback",
    "gsm8k_verifiable_simple",
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
ROBUST_NUMBER_RE = re.compile(
    r"[-+]?(?:\d[\d,]*\s*/\s*\d[\d,]*|\d[\d,]*(?:\.\d+)?|\.\d+)\s*%?"
)
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


def _robust_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    token = _safe_text(value).strip()
    if not token:
        return None
    token = token.replace("$", "").replace(",", "").replace(" ", "").replace("\u00a0", "")
    if token.endswith("%"):
        token = token[:-1]
    if "/" in token:
        try:
            left, right = token.split("/", 1)
            denom = float(right)
            if denom == 0:
                return None
            return float(left) / denom
        except Exception:
            return None
    try:
        return float(token)
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


def _official_extracted_number(response: str) -> str | None:
    match = match_numbers.search(response)
    if match is None:
        return None
    return match.group(1)


def _robust_answer_section(response: str) -> str | None:
    text = _safe_text(response)
    start = text.rfind(solution_start)
    if start < 0:
        return None
    content_start = start + len(solution_start)
    end = text.find(solution_end, content_start)
    if end >= 0:
        return text[content_start:end]
    tail = text[content_start:]
    newline_positions = [pos for pos in (tail.find("\n"), tail.find("\r")) if pos >= 0]
    if newline_positions:
        tail = tail[: min(newline_positions)]
    return tail


def _robust_answer_numbers(response: str) -> list[str]:
    section = _robust_answer_section(response)
    if section is None:
        return []
    return [match.group(0).strip() for match in ROBUST_NUMBER_RE.finditer(section)]


def _robust_any_numbers(response: str) -> list[str]:
    return [match.group(0).strip() for match in ROBUST_NUMBER_RE.finditer(_safe_text(response))]


def _robust_extracted_number(response: str) -> str | None:
    numbers = _robust_answer_numbers(response)
    if not numbers:
        return None
    return numbers[-1]


def _fallback_extracted_number(response: str) -> str | None:
    answer_number = _robust_extracted_number(response)
    if answer_number is not None:
        return answer_number
    numbers = _robust_any_numbers(response)
    if not numbers:
        return None
    return numbers[-1]


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


def _robust_numeric_flags(extracted: str | None, answer: Any) -> tuple[bool, bool]:
    got = _robust_float_or_none(extracted)
    true = _float_or_none(answer)
    if got is None or true is None:
        return False, False
    exact = math.isclose(got, true, rel_tol=0.0, abs_tol=1e-9)
    if true == 0:
        partial = exact
    else:
        partial = abs(got - true) / abs(true) <= 0.10
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


def _numeric_dense_score(response: str, true_answer: Any) -> float:
    if not response.strip():
        return -3.0
    extracted = _robust_extracted_number(response)
    got = _robust_float_or_none(extracted)
    true = _float_or_none(true_answer)
    if got is None or true is None:
        return -2.5
    if math.isclose(got, true, rel_tol=0.0, abs_tol=1e-9):
        return 6.0
    if true == 0:
        return -1.0
    rel_error = abs(got - true) / abs(true)
    if rel_error <= 0.01:
        return 3.0
    if rel_error <= 0.05:
        return 2.0
    if rel_error <= 0.10:
        return 1.0
    if rel_error <= 0.25:
        return 0.25
    return -1.0


def _numeric_guarded_score(response: str, true_answer: Any) -> float:
    if not response.strip():
        return -8.0
    if solution_start not in response:
        return -4.0
    extracted = _robust_extracted_number(response)
    got = _robust_float_or_none(extracted)
    true = _float_or_none(true_answer)
    if got is None or true is None:
        return -4.0
    if math.isclose(got, true, rel_tol=0.0, abs_tol=1e-9):
        return 6.0
    if true == 0:
        return -1.5
    rel_error = abs(got - true) / abs(true)
    if rel_error <= 0.01:
        return 3.0
    if rel_error <= 0.05:
        return 2.0
    if rel_error <= 0.10:
        return 1.0
    if rel_error <= 0.25:
        return 0.25
    return -1.5


def _answer_hygiene_guarded_raw_score(response: str) -> float:
    if not response.strip():
        return 0.0
    starts = response.count(solution_start)
    ends = response.count(solution_end)
    numbers = _robust_answer_numbers(response)
    has_robust_number = bool(numbers)
    score = 0.0
    if has_robust_number:
        if starts == 1:
            score += 0.15
        if ends == 1:
            score += 0.20
        if solution_end in response and not _text_after_solution_close(response):
            score += 0.10
        if len(numbers) == 1:
            score += 0.10
    if solution_start in response and solution_end not in response:
        score -= 0.6
    duplicate_or_broken = starts > 1 or ends > 1 or (starts == 1 and ends == 1 and response.find(solution_start) > response.find(solution_end))
    if duplicate_or_broken:
        score -= 0.6
    return score


def _numeric_guarded_components(response: str, true_answer: Any) -> tuple[float, float, float, float]:
    numeric_score = _numeric_guarded_score(response, true_answer)
    hygiene_raw = _answer_hygiene_guarded_raw_score(response)
    hygiene_effective = min(hygiene_raw, 0.0) if numeric_score < 0 else hygiene_raw
    return numeric_score, hygiene_effective, hygiene_raw, numeric_score + hygiene_effective


def _numeric_score_from_value(got: float | None, true: float | None, *, far_wrong: float) -> float:
    if got is None or true is None:
        return -4.0
    if math.isclose(got, true, rel_tol=0.0, abs_tol=1e-9):
        return 6.0
    if true == 0:
        return far_wrong
    rel_error = abs(got - true) / abs(true)
    if rel_error <= 0.01:
        return 3.0
    if rel_error <= 0.05:
        return 2.0
    if rel_error <= 0.10:
        return 1.0
    if rel_error <= 0.25:
        return 0.25
    return far_wrong


def _fallback_hygiene_raw_score(response: str, *, has_answer_number: bool) -> float:
    if not response.strip():
        return 0.0
    starts = response.count(solution_start)
    ends = response.count(solution_end)
    answer_numbers = _robust_answer_numbers(response)
    score = 0.0
    if has_answer_number:
        if starts == 1:
            score += 0.15
        if ends == 1:
            score += 0.20
        if solution_end in response and not _text_after_solution_close(response):
            score += 0.10
        if len(answer_numbers) == 1:
            score += 0.10
    if starts == 0:
        score -= 0.6
    if starts > 0 and ends == 0:
        score -= 0.6
    duplicate_or_broken = starts > 1 or ends > 1 or (starts == 1 and ends == 1 and response.find(solution_start) > response.find(solution_end))
    if duplicate_or_broken:
        score -= 0.6
    if solution_end in response and _text_after_solution_close(response):
        score -= 0.2
    return score


def _numeric_guarded_fallback_components(response: str, true_answer: Any) -> tuple[float, float, float, float, bool, str | None]:
    """R11: grade the last number anywhere when answer-tag extraction fails.

    This keeps official evaluation unchanged while preventing malformed-but-
    informative completions from collapsing into one identical no-signal bucket.
    """
    if not response.strip():
        return -8.0, 0.0, 0.0, -8.0, False, None
    answer_number = _robust_extracted_number(response)
    fallback_number = None
    used_fallback = False
    extracted = answer_number
    if extracted is None:
        fallback_number = _fallback_extracted_number(response)
        extracted = fallback_number
        used_fallback = extracted is not None

    got = _robust_float_or_none(extracted)
    true = _float_or_none(true_answer)
    if got is None or true is None:
        return -4.0, 0.0, 0.0, -4.0, used_fallback, fallback_number
    numeric_score = _numeric_score_from_value(got, true, far_wrong=-1.5)
    if used_fallback:
        numeric_score -= 0.75
    hygiene_raw = _fallback_hygiene_raw_score(response, has_answer_number=answer_number is not None)
    hygiene_effective = min(hygiene_raw, 0.0) if numeric_score < 0 else hygiene_raw
    return numeric_score, hygiene_effective, hygiene_raw, numeric_score + hygiene_effective, used_fallback, fallback_number


def _gsm8k_simple_numeric_score(response: str, true_answer: Any) -> float:
    """Conservative GSM8K RLVR-style reward: correct high, wrong low, no answer zero."""
    if not response.strip():
        return 0.0
    extracted = _fallback_extracted_number(response)
    got = _robust_float_or_none(extracted)
    true = _float_or_none(true_answer)
    if got is None or true is None:
        return 0.0
    if math.isclose(got, true, rel_tol=0.0, abs_tol=1e-9):
        return 1.0
    if true == 0:
        return 0.1
    rel_error = abs(got - true) / abs(true)
    if rel_error <= 0.01:
        return 0.5
    if rel_error <= 0.10:
        return 0.25
    return 0.1


def _gsm8k_simple_format_score(response: str) -> float:
    """Small non-negative answer-tag helper; never lets format dominate correctness."""
    if not response.strip():
        return 0.0
    starts = response.count(solution_start)
    ends = response.count(solution_end)
    numbers = _robust_answer_numbers(response)
    if starts == 1 and ends == 1 and numbers and not _text_after_solution_close(response):
        return 0.2
    if starts == 1 and numbers:
        return 0.1
    return 0.0


def _answer_hygiene_dense_score(response: str) -> float:
    if not response.strip():
        return 0.0
    starts = response.count(solution_start)
    ends = response.count(solution_end)
    numbers = _robust_answer_numbers(response)
    score = 0.0
    score += 0.3 if starts == 1 else 0.0
    score += 0.2 if ends >= 1 else 0.0
    if len(numbers) == 1:
        score += 0.3
    elif len(numbers) > 1:
        score -= 0.8
    if starts > 1 or ends > 1:
        score -= 0.5
    return score


def _text_after_solution_close(response: str) -> str:
    index = response.find(solution_end)
    if index < 0:
        return ""
    return response[index + len(solution_end) :].strip()


def _closed_answer_minimal_score(response: str) -> float:
    """Minimal answer-tag shaping for R9; numeric reward stays dominant."""
    if not response.strip():
        return 0.0
    starts = response.count(solution_start)
    ends = response.count(solution_end)
    section = _robust_answer_section(response)
    numbers = _robust_answer_numbers(response)
    score = 0.0
    if starts == 1:
        score += 0.2
    if ends == 1:
        score += 0.4
    if section is not None and section.strip():
        score += 0.2
    if solution_end in response and _text_after_solution_close(response):
        score -= 0.3
    if solution_start in response and solution_end not in response:
        score -= 0.4
    if starts > 1 or ends > 1:
        score -= 0.5
    if len(numbers) == 1:
        score += 0.1
    elif len(numbers) > 1:
        score -= 0.2
    return score


def _length_penalty_short(response: str) -> float:
    chars = len(response)
    if chars <= 1000:
        penalty = 0.0
    elif chars <= 1600:
        penalty = -0.25
    else:
        penalty = -0.75
    if solution_end not in response and chars > 1200:
        penalty -= 0.5
    return penalty


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


def numeric_dense(prompts, completions, answer, **kwargs):
    """Dense numeric reward using the robust last-number answer extractor."""
    return [_numeric_dense_score(response, true) for response, true in zip(completions, answer)]


def answer_hygiene_dense(prompts, completions, **kwargs):
    """Small answer-only hygiene reward for the dense numeric variants."""
    return [_answer_hygiene_dense_score(response) for response in completions]


def length_penalty_1200(prompts, completions, **kwargs):
    """Mild penalty for responses longer than 1200 characters."""
    return [_length_penalty_len1200(response) for response in completions]


def length_penalty_short(prompts, completions, **kwargs):
    """Very light length/no-close penalty for the short dense numeric variant."""
    return [_length_penalty_short(response) for response in completions]


def closed_answer_minimal(prompts, completions, **kwargs):
    """Minimal answer closing/tag hygiene reward for the targeted R9 run."""
    return [_closed_answer_minimal_score(response) for response in completions]


def numeric_guarded_total(prompts, completions, answer, **kwargs):
    """R10 guarded numeric reward with gated answer hygiene."""
    return [_numeric_guarded_components(response, true)[3] for response, true in zip(completions, answer)]


def numeric_guarded_fallback_total(prompts, completions, answer, **kwargs):
    """R11 guarded numeric reward with whole-completion last-number fallback."""
    return [_numeric_guarded_fallback_components(response, true)[3] for response, true in zip(completions, answer)]


def gsm8k_simple_numeric(prompts, completions, answer, **kwargs):
    """R12 simple GSM8K verifiable reward with tolerant final-number extraction."""
    return [_gsm8k_simple_numeric_score(response, true) for response, true in zip(completions, answer)]


def gsm8k_simple_format(prompts, completions, **kwargs):
    """Small answer-tag reward for the simple verifiable branch."""
    return [_gsm8k_simple_format_score(response) for response in completions]


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
        official_extracted_number = _official_extracted_number(completion)
        official_numeric_exact, official_numeric_partial = _numeric_flags(official_extracted_number, answer)
        robust_extracted_number = _robust_extracted_number(completion)
        robust_numeric_exact, robust_numeric_partial = _robust_numeric_flags(robust_extracted_number, answer)
        tag_counts = _tag_counts(completion)
        answer_numbers = NUMBER_RE.findall(_answer_section(completion) or "")
        robust_answer_numbers = _robust_answer_numbers(completion)

        numeric_primary_score = _numeric_primary_score(completion, answer)
        numeric_dense_score = _numeric_dense_score(completion, answer)
        format_strict_light_score = _strict_light_score(completion)
        answer_tag_light_score = _answer_tag_light_score(completion)
        answer_hygiene_score = _answer_hygiene_dense_score(completion)
        closed_answer_score = _closed_answer_minimal_score(completion)
        numeric_guarded_score, answer_hygiene_guarded_score, answer_hygiene_guarded_raw_score, numeric_guarded_total_score = (
            _numeric_guarded_components(completion, answer)
        )
        (
            numeric_guarded_fallback_score,
            answer_hygiene_fallback_score,
            answer_hygiene_fallback_raw_score,
            numeric_guarded_fallback_total_score,
            fallback_number_used,
            fallback_extracted_number,
        ) = _numeric_guarded_fallback_components(completion, answer)
        fallback_numeric_exact, fallback_numeric_partial = _robust_numeric_flags(fallback_extracted_number, answer)
        gsm8k_simple_numeric_score = _gsm8k_simple_numeric_score(completion, answer)
        gsm8k_simple_format_score = _gsm8k_simple_format_score(completion)
        length_penalty_score = _length_penalty_len1200(completion)
        length_penalty_short_score = _length_penalty_short(completion)
        format_light_total = format_strict_light_score + answer_tag_light_score

        component_values = {
            "match_format_exactly": baseline["match_format_exactly"],
            "match_format_approximately": baseline["match_format_approximately"],
            "check_answer": baseline["check_answer"],
            "check_numbers": baseline["check_numbers"],
            "format_strict_light": format_strict_light_score,
            "answer_tag_light": answer_tag_light_score,
            "numeric_primary": numeric_primary_score,
            "numeric_dense": numeric_dense_score,
            "answer_hygiene_dense": answer_hygiene_score,
            "closed_answer_minimal": closed_answer_score,
            "numeric_guarded": numeric_guarded_score,
            "answer_hygiene_guarded": answer_hygiene_guarded_score,
            "answer_hygiene_guarded_raw": answer_hygiene_guarded_raw_score,
            "numeric_guarded_total": numeric_guarded_total_score,
            "numeric_guarded_fallback": numeric_guarded_fallback_score,
            "answer_hygiene_fallback": answer_hygiene_fallback_score,
            "answer_hygiene_fallback_raw": answer_hygiene_fallback_raw_score,
            "numeric_guarded_fallback_total": numeric_guarded_fallback_total_score,
            "gsm8k_simple_numeric": gsm8k_simple_numeric_score,
            "gsm8k_simple_format": gsm8k_simple_format_score,
            "length_penalty_1200": length_penalty_score,
            "length_penalty_short": length_penalty_short_score,
        }
        mode_components = reward_components_for_mode(mode)
        mode_reward = float(sum(component_values[name] for name in mode_components))
        parser_false_negative = bool(robust_numeric_exact and not official_numeric_exact)

        rows.append(
            {
                **baseline,
                "reward_mode": mode,
                "reward_components_for_mode": mode_components,
                "reward_total_recomputed": mode_reward,
                "format_strict_light": format_strict_light_score,
                "answer_tag_light": answer_tag_light_score,
                "numeric_primary": numeric_primary_score,
                "numeric_dense": numeric_dense_score,
                "answer_hygiene_dense": answer_hygiene_score,
                "closed_answer_minimal": closed_answer_score,
                "numeric_guarded": numeric_guarded_score,
                "answer_hygiene_guarded": answer_hygiene_guarded_score,
                "answer_hygiene_guarded_raw": answer_hygiene_guarded_raw_score,
                "numeric_guarded_total": numeric_guarded_total_score,
                "numeric_guarded_fallback": numeric_guarded_fallback_score,
                "answer_hygiene_fallback": answer_hygiene_fallback_score,
                "answer_hygiene_fallback_raw": answer_hygiene_fallback_raw_score,
                "numeric_guarded_fallback_total": numeric_guarded_fallback_total_score,
                "gsm8k_simple_numeric": gsm8k_simple_numeric_score,
                "gsm8k_simple_format": gsm8k_simple_format_score,
                "length_penalty_1200": length_penalty_score,
                "length_penalty_short": length_penalty_short_score,
                "format_light_total": format_light_total,
                "extracted_number": extracted_number,
                "official_extracted_number": official_extracted_number,
                "robust_extracted_number": robust_extracted_number,
                "fallback_extracted_number": fallback_extracted_number,
                "numeric_exact": numeric_exact,
                "numeric_partial": numeric_partial,
                "official_numeric_exact": official_numeric_exact,
                "official_numeric_partial": official_numeric_partial,
                "robust_numeric_exact": robust_numeric_exact,
                "robust_numeric_partial": robust_numeric_partial,
                "fallback_numeric_exact": fallback_numeric_exact,
                "fallback_numeric_partial": fallback_numeric_partial,
                "fallback_number_used": fallback_number_used,
                "parser_false_negative": parser_false_negative,
                "answer_tag_pair_ok": _answer_tag_pair_ok(completion),
                "duplicate_or_broken_answer_tag": _duplicate_or_broken_answer_tag(completion),
                "overlong_1200": len(completion) > 1200,
                "overlong_1600": len(completion) > 1600,
                "answer_multi_number": len(answer_numbers) > 1,
                "answer_single_number": len(robust_answer_numbers) == 1,
                "robust_answer_number_count": len(robust_answer_numbers),
                "no_close_answer": solution_start in completion and solution_end not in completion,
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
    if mode == "numeric_dense_lastnum":
        return ["numeric_dense"]
    if mode == "numeric_dense_single_answer":
        return ["numeric_dense", "answer_hygiene_dense"]
    if mode == "numeric_dense_single_answer_short":
        return ["numeric_dense", "answer_hygiene_dense", "length_penalty_short"]
    if mode == "closed_answer_minimal":
        return ["numeric_dense", "closed_answer_minimal"]
    if mode == "numeric_guarded":
        return ["numeric_guarded", "answer_hygiene_guarded"]
    if mode == "numeric_guarded_fallback":
        return ["numeric_guarded_fallback", "answer_hygiene_fallback"]
    if mode == "gsm8k_verifiable_simple":
        return ["gsm8k_simple_numeric", "gsm8k_simple_format"]
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
    if mode == "numeric_dense_lastnum":
        return [numeric_dense]
    if mode == "numeric_dense_single_answer":
        return [numeric_dense, answer_hygiene_dense]
    if mode == "numeric_dense_single_answer_short":
        return [numeric_dense, answer_hygiene_dense, length_penalty_short]
    if mode == "closed_answer_minimal":
        return [numeric_dense, closed_answer_minimal]
    if mode == "numeric_guarded":
        return [numeric_guarded_total]
    if mode == "numeric_guarded_fallback":
        return [numeric_guarded_fallback_total]
    if mode == "gsm8k_verifiable_simple":
        return [gsm8k_simple_numeric, gsm8k_simple_format]
    raise ValueError(f"Unknown REWARD_MODE '{mode}'. Valid modes: {', '.join(REWARD_MODES)}")


REWARD_FNS = reward_functions_for_mode(REWARD_MODE)
