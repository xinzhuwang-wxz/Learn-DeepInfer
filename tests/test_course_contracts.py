"""Executable whole-course and source-reading contracts, not device tests."""
import sys
from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'examples'))
from check_system import check
from audit_evidence import audit
from s11_read_state import walk
from observe_lesson import run_case

class CourseContracts(unittest.TestCase):
    def test_integrated(self):
        self.assertEqual(check()['final_report']['committed_characters'], 13)
    def test_source_reading(self):
        rows = walk()
        self.assertEqual(rows[2][1], [0,1,0])
        self.assertEqual(rows[3][1], [0,0,0])
        self.assertNotEqual(rows[0][1], rows[3][1])
    def test_hashes_without_execution(self):
        self.assertEqual(audit()['hashes_and_ranges'], 'passed')
    def test_progress_tables(self):
        expected = [5,9,9,16,14,15,15,15,15]
        for stage, ticks in enumerate(expected, 1):
            engine, output, rows = run_case(stage)
            self.assertEqual(engine.ticks, ticks)
            if stage >= 2:
                for row in rows:
                    self.assertTrue(all(0 <= got <= total for got,total in zip(row['computed'], [7,3,5])))
                self.assertEqual(rows[-1]['computed'], [7,3,5])
    def test_source_marked_excerpts(self):
        sys.path.insert(0, str(ROOT/'scripts'))
        from check_snippets import check as snippets
        errors, entries = snippets((ROOT/'docs').glob('*.md'))
        self.assertEqual(errors, [])
        self.assertGreater(len(entries), 70)
