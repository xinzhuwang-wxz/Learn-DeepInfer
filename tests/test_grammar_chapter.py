"""Assertions for chapter 08's actual method boundaries and failure limits."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from s08_grammar_walkthrough import observe, rollback_case
from observe_lesson import load


class GrammarChapter(unittest.TestCase):
    def test_committed_boundaries(self):
        engine, rows = observe()
        for row in rows:
            self.assertEqual(row['outputs'], row['prefixes'])
        by_tick = {r['tick']: r for r in rows}
        self.assertEqual(by_tick[13]['prefixes'][2], '')
        self.assertEqual(by_tick[13]['pending'], [('B', 'e'), ('C', 'tr')])
        self.assertEqual(by_tick[14]['prefixes'][2], 'tr')
        self.assertEqual(by_tick[14]['pending'], [('C', 'ue')])
        self.assertEqual(by_tick[15]['prefixes'][2], 'true')
        self.assertEqual(by_tick[15]['live_refs'], 0)
        self.assertEqual(engine.report()['committed_characters'], 13)

    def test_restore_and_reject(self):
        self.assertEqual(rollback_case()[-1], ('rollback(2)', None, 't', ['r']))
        matcher = load(8).Matcher()
        matcher.accept('t')
        self.assertFalse(matcher.accept('x'))
        self.assertEqual(matcher.prefix, 't')
        matcher.rollback(0)
        self.assertEqual(matcher.prefix, 't')
        for steps in (-1, 2):
            with self.assertRaises(ValueError): matcher.rollback(steps)
            self.assertEqual(matcher.prefix, 't')

    def test_partial_chunk_is_not_transactional(self):
        lesson = load(8)
        request = lesson.Request('X', 1, 'tx')
        engine = lesson.Engine([request])
        with self.assertRaises(ValueError): engine.run()
        self.assertEqual(request.output, '')
        self.assertEqual(request.matcher.prefix, 't')
        self.assertEqual(sum(engine.pool.refs), 0)
        self.assertEqual(engine.pending, [])
        self.assertNotIn('X', engine.finished)

    def test_eos_is_not_stored(self):
        matcher = load(8).Matcher()
        for token in 'true': self.assertTrue(matcher.accept(token))
        self.assertTrue(matcher.accept('<eos>'))
        self.assertTrue(matcher.accept('<eos>'))
        matcher.rollback(1)
        self.assertEqual(matcher.prefix, 'tru')
        self.assertEqual(matcher.allowed(), {'e'})


if __name__ == '__main__':
    unittest.main()
