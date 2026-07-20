import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_qwenlm import (
    extract_numeric_answer,
    extract_predicted_option,
    grade_deterministic,
    grade_mcq_option,
    grade_numeric_answer,
)


class McqOptionTest(unittest.TestCase):
    def test_labeled_answer_does_not_use_answer_initial(self):
        self.assertEqual(extract_predicted_option("Answer: B"), "B")
        self.assertEqual(extract_predicted_option("The final answer is (C)."), "C")

    def test_common_direct_answer_formats(self):
        for answer in ("D", "(D)", "D.", "Option D", "choice: d"):
            self.assertEqual(extract_predicted_option(answer), "D")

    def test_parseable_wrong_answer_is_deterministic(self):
        self.assertTrue(grade_mcq_option("B", "Answer: B"))
        self.assertFalse(grade_mcq_option("A", "Answer: B"))
        self.assertEqual(grade_deterministic("vstar", "A", "Answer: B"), ("No", "mcq_option"))

    def test_unparseable_answer_uses_fallback(self):
        self.assertIsNone(grade_mcq_option("A", "I cannot determine the answer."))

    def test_numeric_answers(self):
        self.assertEqual(extract_numeric_answer("There are 11."), 11)
        self.assertTrue(grade_numeric_answer("4", "There are 4."))
        self.assertFalse(grade_numeric_answer("4", "5"))
        self.assertEqual(grade_deterministic("zoombench", "4", "5"), ("No", "numeric_exact"))
        self.assertIsNone(grade_numeric_answer("A", "A"))
        self.assertIsNone(grade_numeric_answer("4", "between 4 and 5"))


if __name__ == "__main__":
    unittest.main()
