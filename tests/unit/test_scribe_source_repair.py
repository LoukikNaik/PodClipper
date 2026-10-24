"""Full-source speech repairs are not limited to the short-reel retry budget."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import wave
import numpy as np

from podclipper.config import load_default_config
from podclipper.scribe import ScribeTranscriber
from podclipper.types import Transcript, TranscriptSegment, Word
from podclipper.transcribe_repair import CaptionCoverageError
from podclipper.analyze import analyze_for_reels


class SourceRepairTests(unittest.TestCase):
    def exercise_unresolved_source(self, directory, speech):
        cfg = load_default_config()
        cfg.llm.endpoint = 'https://openrouter.ai/api/v1'
        audio = directory / 'audio.wav'
        # Encode the source time in the sample amplitude so the fake external
        # transcriber can return words relative to each real repair window.
        samples = (np.arange(300 * 16000) / 16000 / 1000 * 32768).astype('<i2')
        with wave.open(str(audio), 'wb') as output:
            output.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            output.writeframes(samples.tobytes())
        with patch.dict(os.environ, {'LLM_API_KEY': 'test-only'}):
            transcriber = ScribeTranscriber(cfg, directory)

        def transcribe(samples):
            duration = len(samples) / 16000
            if duration > 299:
                return [Word(0, .4, 'start'), Word(299, 299.4, 'end')], 'en'
            offset = float(samples[0]) * 1000
            return [Word(float(t - offset), float(t - offset + .4), 'repaired')
                    for t in np.arange(30, 40, .5) if offset <= t and t + .4 <= offset + duration], 'en'

        with patch.dict(os.environ, {'LLM_API_KEY': 'test-only'}), patch.object(
                transcriber, '_request', side_effect=transcribe), patch(
                'podclipper.transcribe_repair._speech_regions', return_value=speech):
            transcript = transcriber.full_transcript(audio, 300)
        return transcriber, transcript, audio

    def test_podcast_excludes_small_unrepaired_gap_and_keeps_successful_repairs(self):
        with tempfile.TemporaryDirectory() as temporary:
            transcriber, transcript, audio = self.exercise_unresolved_source(Path(temporary), [(10, 12), (30, 40)])
            self.assertTrue(any(w.text == 'repaired' for s in transcript.segments for w in s.words))
            self.assertEqual(len(transcriber.unreliable_ranges), 1)
            left, right = transcriber.unreliable_ranges[0]
            self.assertLessEqual(left, 10)
            self.assertGreaterEqual(right, 12)
            self.assertLess(right - left, 30)
            with patch.object(transcriber, '_request', side_effect=AssertionError('must reuse cache')):
                transcriber.full_transcript(audio, 300)
            self.assertEqual(transcriber.unreliable_ranges, [[left, right]])

    def test_large_unrepaired_source_failure_still_stops_processing(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(CaptionCoverageError):
                self.exercise_unresolved_source(Path(temporary), [(100, 150)])

    def test_model_cannot_select_a_clip_overlapping_an_excluded_section(self):
        cfg = load_default_config()
        transcript = Transcript('en', [TranscriptSegment(0, 100, 'A complete conversation.',
                                [Word(i, i + .4, 'speech') for i in range(100)])])
        provider = Mock(name='external-model')
        provider.name = 'test'
        provider.complete.return_value = json.dumps([
            {'start': 20, 'end': 40, 'title': 'A Complete Conversation', 'hook_score': .9}])
        result = analyze_for_reels(transcript, 100, provider, cfg, excluded_ranges=[[25, 30]])
        self.assertEqual(result, [])
        self.assertIn('No selected clip may overlap', provider.complete.call_args.kwargs['user_prompt'])

    def test_exported_clip_with_missing_speech_still_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            transcriber, transcript, _ = self.exercise_unresolved_source(directory, [(10, 12), (30, 40)])
            segment = directory / 'clip.mp4'
            segment.touch()
            with patch('podclipper.transcribe._decode_audio_to_float32', return_value=np.zeros(20 * 16000)), patch(
                    'podclipper.transcribe_repair._speech_regions', return_value=[(10, 12)]), patch.object(
                    transcriber, '_request', return_value=([], 'en')):
                with self.assertRaises(CaptionCoverageError):
                    transcriber.clip_words(segment, transcript, 0, 20, directory)

    def test_five_minute_chunk_can_repair_more_than_eight_sections(self):
        cfg = load_default_config()
        cfg.llm.endpoint = 'https://openrouter.ai/api/v1'
        cfg.transcribe.scribe.source_repair_max_windows = 64
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {'LLM_API_KEY': 'test-only'}):
            directory = Path(temporary)
            audio = directory / 'audio.wav'
            with wave.open(str(audio), 'wb') as output:
                output.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
                output.writeframes(b'\0\0' * (300 * 16000))
            transcriber = ScribeTranscriber(cfg, directory)
            calls = []

            def transcribe(samples):
                duration = len(samples) / 16000
                calls.append(duration)
                if duration > 299:
                    return [Word(0, .4, 'start')], 'en'
                return [Word(i / 2, min(i / 2 + .4, duration), 'speech')
                        for i in range(int(duration * 2))], 'en'

            with patch.object(transcriber, '_request', side_effect=transcribe), patch(
                    'podclipper.transcribe_repair._speech_regions', return_value=[(0, 300)]):
                result = transcriber.full_transcript(audio, 300)
            report = json.loads(next(transcriber.cache.glob('*/chunk-0000.quality.json')).read_text())
            self.assertGreater(len(report['retries']), 8)
            self.assertEqual(report['remaining_issues'], [])
            self.assertEqual(report['status'], 'repaired')
            self.assertGreater(result.segments[-1].end, 298)
            self.assertEqual(cfg.transcribe.caption_repair.max_windows, 8)


if __name__ == '__main__':
    unittest.main()
