"""Website results include every evaluated reel, regardless of verdict."""

from pathlib import Path
import tempfile
import unittest

from podclipper_web.results import finished_reel_path, finished_reels, reel_quality


class WebsiteResultsTests(unittest.TestCase):
    def test_skipped_reel_is_a_finished_result(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            reel = directory / "reel_01.mp4"
            for verdict in ("skip", "review", "publish"):
                self.assertEqual(finished_reel_path(f"[1/3] Done ({verdict}): {reel}", directory), reel.resolve())
            self.assertIsNone(finished_reel_path(f"Subtitle burn complete → {reel}", directory))
            self.assertIsNone(finished_reel_path("[1/3] Done (skip): /outside.mp4", directory))

    def test_quality_verdict_is_available_for_website(self):
        with tempfile.TemporaryDirectory() as root:
            reel = Path(root) / "reel.mp4"
            self.assertIsNone(reel_quality(reel))
            reel.with_suffix(".txt").write_text('title: Example\nOverall:               0.61  →  SKIP\nFeedback:              "Weak hook"\n')
            self.assertEqual(reel_quality(reel), "skip")

    def test_old_running_job_exposes_skipped_export_only_after_evaluation(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root).resolve()
            output = directory / "reels" / "run"
            output.mkdir(parents=True)
            reel = output / "reel.mp4"
            reel.touch()
            reel.with_suffix(".captions.json").write_text('{}')
            reel.with_suffix(".txt").write_text('title: Example\n')
            self.assertEqual(finished_reels(directory, {}), [])
            reel.with_suffix(".txt").write_text('title: Example\nOverall: 0.61 → SKIP\nFeedback: "Weak hook"\n')
            self.assertEqual(finished_reels(directory, {}), [str(reel)])


if __name__ == "__main__":
    unittest.main()
