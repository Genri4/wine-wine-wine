import unittest

from recognition.confidence import decide_unknown


class ConfidenceTests(unittest.TestCase):
    def test_score_below_threshold_is_unknown(self) -> None:
        decision = decide_unknown(0.49, 0.5)
        self.assertTrue(decision.is_unknown)

    def test_score_at_threshold_is_known(self) -> None:
        decision = decide_unknown(0.5, 0.5)
        self.assertFalse(decision.is_unknown)

    def test_missing_score_is_unknown(self) -> None:
        self.assertTrue(decide_unknown(None, 0.5).is_unknown)


if __name__ == "__main__":
    unittest.main()
