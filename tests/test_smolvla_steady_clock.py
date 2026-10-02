from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SmolVLATimingClockTests(unittest.TestCase):
    def test_predict_uses_one_monotonic_clock_for_all_phase_and_total_samples(self):
        source = (ROOT / "models" / "smolvla.cpp").read_text(encoding="utf-8")
        signature = "std::vector<float> SmolVLAModelArch::predict(const Inputs& in) {"
        start = source.index(signature)
        end = source.index("\n}\n", start) + 2
        predict = source[start:end]

        self.assertIn("using clk = std::chrono::steady_clock;", predict)
        self.assertNotIn("high_resolution_clock", predict)
        self.assertIn("stats.ms_vision =", predict)
        self.assertIn("stats.ms_inference =", predict)
        self.assertIn("stats.ms_total =", predict)


if __name__ == "__main__":
    unittest.main()
