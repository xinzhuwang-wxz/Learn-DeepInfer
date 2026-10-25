"""Assertions at stage-06 method boundaries, matching the chapter walkthrough."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'async_walkthrough_tests', ROOT / 'examples/s06_async_walkthrough.py')
walkthrough = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = walkthrough
spec.loader.exec_module(walkthrough)


class AsyncChapter(unittest.TestCase):
    def test_whole_chapter_boundary_table(self):
        engine, _ = walkthrough.run_case()
        expected = [
            (7, ('', '', ''), [('A', 'tr')], 5),
            (8, ('tr', '', ''), [('A', 'ue')], 5),
            (9, ('tr', '', ''), [('A', 'ue')], 8),
            (10, ('true', '', ''), [('B', 'fa')], 5),
            (11, ('true', 'fa', ''), [('B', 'ls')], 5),
            (12, ('true', 'fa', ''), [('B', 'ls')], 5),
            (13, ('true', 'fals', ''), [('B', 'e'), ('C', 'tr')], 5),
            (14, ('true', 'false', 'tr'), [('C', 'ue')], 3),
            (15, ('true', 'false', 'true'), [], 0),
        ]
        for tick, outputs, pending, refs in expected:
            row = next(row for row in engine.snapshots
                       if row['tick'] == tick and row['boundary'].startswith('after-'))
            with self.subTest(tick=tick):
                self.assertEqual(tuple(row['output'][key] for key in ('A', 'B', 'C')), outputs)
                self.assertEqual(row['pending'], pending)
                self.assertEqual(row['live_refs'], refs)

    def test_before_after_and_repeated_flush(self):
        engine, result = walkthrough.run_case()
        self.assertEqual(result, ['true', 'false', 'true'])
        before = next(row for row in engine.snapshots if row['boundary'] == 'before-flush')
        self.assertEqual(before['output']['C'], 'tr')
        self.assertEqual(before['pending'], [('C', 'ue')])
        self.assertEqual(before['live_refs'], 3)
        self.assertEqual(engine.snapshots[-1]['finished'], ['A', 'B', 'C'])

    def test_next_prefill_keeps_pending_in_this_teaching_model(self):
        engine, _ = walkthrough.run_case()
        row = next(row for row in engine.snapshots if row['tick'] == 9)
        self.assertEqual(row['boundary'], 'after-prefill')
        self.assertEqual(row['pending'], [('A', 'ue')])
        self.assertEqual(row['live_refs'], 8)

    def test_request_identity_survives_active_batch_change(self):
        engine, _ = walkthrough.run_case()
        start = engine.trace.index(('sample-pending', 'C', 'tr')) + 1
        # Phase negotiation precedes decode; the chapter starts at forward.
        tail = [event for event in engine.trace[start:] if event[0] != 'phase']
        self.assertEqual(tail[:7], [
            ('forward', ['C']), ('emit', 'B', 'e'), ('emit', 'C', 'tr'),
            ('verify-plan', [2]), ('sample-pending', 'C', 'ue'),
            ('flush', 1), ('emit', 'C', 'ue'),
        ])


if __name__ == '__main__':
    unittest.main()
