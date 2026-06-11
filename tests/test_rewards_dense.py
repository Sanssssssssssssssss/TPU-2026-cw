import os
import sys
import types
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

data_stub = types.ModuleType("data")
data_stub.reasoning_start = "<reasoning>"
data_stub.reasoning_end = "</reasoning>"
data_stub.solution_start = "<answer>"
data_stub.solution_end = "</answer>"
sys.modules.setdefault("data", data_stub)

import rewards  # noqa: E402

solution_start = data_stub.solution_start
solution_end = data_stub.solution_end


def answer(text: str) -> str:
    return f"{solution_start}{text}{solution_end}"


class DenseNumericRewardTests(unittest.TestCase):
    def test_official_parser_remains_first_number_but_robust_takes_last(self):
        completion = answer("378 * 4 = 1512")

        official = rewards.match_numbers.search(completion).group(1)
        robust = rewards._robust_extracted_number(completion)

        self.assertEqual(official, "378")
        self.assertEqual(robust, "1512")
        self.assertEqual(rewards._numeric_dense_score(completion, "1512"), 6.0)

    def test_robust_number_formats(self):
        cases = {
            "1,512": 1512.0,
            "$1512": 1512.0,
            "-3.5": -3.5,
            "6.00": 6.0,
            "1/4": 0.25,
            "50%": 50.0,
        }
        for token, expected in cases.items():
            with self.subTest(token=token):
                completion = answer(token)
                extracted = rewards._robust_extracted_number(completion)
                self.assertEqual(rewards._robust_float_or_none(extracted), expected)

    def test_dense_reward_ladder(self):
        self.assertEqual(rewards._numeric_dense_score(answer("100"), "100"), 6.0)
        self.assertEqual(rewards._numeric_dense_score(answer("101"), "100"), 3.0)
        self.assertEqual(rewards._numeric_dense_score(answer("105"), "100"), 2.0)
        self.assertEqual(rewards._numeric_dense_score(answer("110"), "100"), 1.0)
        self.assertEqual(rewards._numeric_dense_score(answer("125"), "100"), 0.25)
        self.assertEqual(rewards._numeric_dense_score(answer("130"), "100"), -1.0)
        self.assertEqual(rewards._numeric_dense_score(answer("no number"), "100"), -2.5)
        self.assertEqual(rewards._numeric_dense_score("", "100"), -3.0)

    def test_answer_hygiene_single_number_and_multi_number(self):
        self.assertAlmostEqual(rewards._answer_hygiene_dense_score(answer("1512")), 0.8)
        self.assertAlmostEqual(rewards._answer_hygiene_dense_score(answer("378 * 4 = 1512")), -0.3)

        duplicated = f"{solution_start}1{solution_end} {solution_start}2{solution_end}"
        self.assertAlmostEqual(rewards._answer_hygiene_dense_score(duplicated), 0.0)

    def test_short_length_penalty(self):
        closed_long = answer("1" * 1001)
        self.assertEqual(rewards._length_penalty_short(closed_long), -0.25)

        overlong = answer("1" * 1601)
        self.assertEqual(rewards._length_penalty_short(overlong), -0.75)

        no_close = solution_start + ("1" * 1201)
        self.assertEqual(rewards._length_penalty_short(no_close), -0.75)

    def test_closed_answer_minimal_score(self):
        self.assertAlmostEqual(rewards._closed_answer_minimal_score(answer("1512")), 0.9)

        multi = answer("378 * 4 = 1512")
        self.assertAlmostEqual(rewards._closed_answer_minimal_score(multi), 0.6)

        no_close = solution_start + "1512"
        self.assertAlmostEqual(rewards._closed_answer_minimal_score(no_close), 0.1)

        trailing = answer("1512") + " extra"
        self.assertAlmostEqual(rewards._closed_answer_minimal_score(trailing), 0.6)

        duplicated = f"{solution_start}1{solution_end} {solution_start}2{solution_end}"
        self.assertAlmostEqual(rewards._closed_answer_minimal_score(duplicated), -0.5)

    def test_numeric_guarded_empty_and_wrong_formatted(self):
        self.assertEqual(rewards._numeric_guarded_components("", "100")[3], -8.0)

        wrong_clean = answer("130")
        numeric, hygiene, hygiene_raw, total = rewards._numeric_guarded_components(wrong_clean, "100")
        self.assertEqual(numeric, -1.5)
        self.assertAlmostEqual(hygiene_raw, 0.55)
        self.assertEqual(hygiene, 0.0)
        self.assertEqual(total, -1.5)

    def test_numeric_guarded_missing_close_and_exact(self):
        no_close = solution_start + "100"
        numeric, hygiene, hygiene_raw, total = rewards._numeric_guarded_components(no_close, "100")
        self.assertEqual(numeric, 6.0)
        self.assertAlmostEqual(hygiene_raw, -0.35)
        self.assertAlmostEqual(hygiene, -0.35)
        self.assertAlmostEqual(total, 5.65)

        numeric, hygiene, hygiene_raw, total = rewards._numeric_guarded_components(answer("100"), "100")
        self.assertEqual(numeric, 6.0)
        self.assertAlmostEqual(hygiene_raw, 0.55)
        self.assertAlmostEqual(hygiene, 0.55)
        self.assertAlmostEqual(total, 6.55)

    def test_numeric_guarded_no_answer_or_number(self):
        self.assertEqual(rewards._numeric_guarded_components("100", "100")[3], -4.0)
        self.assertEqual(rewards._numeric_guarded_components(answer("no number"), "100")[3], -4.0)

    def test_numeric_guarded_fallback_uses_last_number_outside_answer(self):
        numeric, hygiene, hygiene_raw, total, used, extracted = rewards._numeric_guarded_fallback_components(
            "We calculate 378 * 4 = 1512.", "1512"
        )
        self.assertTrue(used)
        self.assertEqual(extracted, "1512")
        self.assertAlmostEqual(numeric, 5.25)
        self.assertAlmostEqual(hygiene_raw, -0.6)
        self.assertAlmostEqual(hygiene, -0.6)
        self.assertAlmostEqual(total, 4.65)

    def test_numeric_guarded_fallback_keeps_no_number_bucket(self):
        self.assertEqual(rewards._numeric_guarded_fallback_components("No numeric answer here.", "100")[3], -4.0)
        self.assertEqual(rewards._numeric_guarded_fallback_components("", "100")[3], -8.0)

    def test_numeric_guarded_fallback_wrong_format_stays_negative(self):
        numeric, hygiene, hygiene_raw, total, used, extracted = rewards._numeric_guarded_fallback_components(
            answer("130"), "100"
        )
        self.assertFalse(used)
        self.assertEqual(extracted, None)
        self.assertEqual(numeric, -1.5)
        self.assertAlmostEqual(hygiene_raw, 0.55)
        self.assertEqual(hygiene, 0.0)
        self.assertEqual(total, -1.5)

    def test_gsm8k_simple_verifiable_reward(self):
        self.assertEqual(rewards._gsm8k_simple_numeric_score(answer("100"), "100"), 1.0)
        self.assertEqual(rewards._gsm8k_simple_numeric_score(answer("101"), "100"), 0.5)
        self.assertEqual(rewards._gsm8k_simple_numeric_score(answer("110"), "100"), 0.25)
        self.assertEqual(rewards._gsm8k_simple_numeric_score(answer("130"), "100"), 0.1)
        self.assertEqual(rewards._gsm8k_simple_numeric_score("No number here.", "100"), 0.0)

        self.assertEqual(rewards._gsm8k_simple_numeric_score("The answer is 100.", "100"), 1.0)
        self.assertAlmostEqual(rewards._gsm8k_simple_format_score(answer("100")), 0.2)
        self.assertAlmostEqual(rewards._gsm8k_simple_format_score(solution_start + "100"), 0.1)
        self.assertEqual(rewards._gsm8k_simple_format_score("The answer is 100."), 0.0)


if __name__ == "__main__":
    unittest.main()
