import os
import sys
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import evaluate  # noqa: E402


def diag(
    *,
    official_exact=False,
    official_partial=False,
    robust_exact=False,
    format_ok=False,
    official="1",
    robust="1",
    count=1,
):
    return {
        "official_extracted_number": official,
        "robust_extracted_number": robust,
        "official_numeric_exact": official_exact,
        "official_numeric_partial": official_partial,
        "robust_numeric_exact": robust_exact,
        "robust_numeric_partial": robust_exact,
        "format_ok": format_ok,
        "robust_answer_number_count": count,
        "parser_false_negative": bool(robust_exact and not official_exact),
    }


class EvaluateExampleTaxonomyTests(unittest.TestCase):
    def test_correct(self):
        completion = "<reasoning>x</reasoning><answer>1512</answer>"
        self.assertEqual(
            evaluate.classify_failure(completion, diag(official_exact=True, format_ok=True)),
            "correct",
        )

    def test_missing_close(self):
        completion = "<answer>1512"
        self.assertEqual(evaluate.classify_failure(completion, diag()), "missing_answer_close")

    def test_trailing_text_after_close(self):
        completion = "<answer>1512</answer> trailing"
        self.assertEqual(
            evaluate.classify_failure(completion, diag(official_exact=True, format_ok=False)),
            "trailing_text_after_close",
        )

    def test_multiple_numbers(self):
        completion = "<answer>378 * 4 = 1512</answer>"
        self.assertEqual(
            evaluate.classify_failure(
                completion,
                diag(official_exact=False, robust_exact=True, official="378", robust="1512", count=2),
            ),
            "multiple_numbers",
        )

    def test_empty(self):
        self.assertEqual(evaluate.classify_failure("", diag(official=None, robust=None, count=0)), "empty_completion")

    def test_wrong_number(self):
        completion = "<answer>13</answer>"
        self.assertEqual(evaluate.classify_failure(completion, diag(official_exact=False)), "wrong_number")


if __name__ == "__main__":
    unittest.main()
