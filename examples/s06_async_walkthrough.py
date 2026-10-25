"""Observe stage 06 without changing its scheduler or output logic.

Standard-library CPU teaching run. No model, GPU, network, or performance timing.
Run from the repository root: python examples/s06_async_walkthrough.py
"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import sys

SOURCE = Path(__file__).resolve().parents[1] / 'lessons/06_async_lifetime/code.py'
spec = importlib.util.spec_from_file_location('mini_infer_async_chapter', SOURCE)
lesson = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = lesson
spec.loader.exec_module(lesson)


class ObservedEngine(lesson.Engine):
    """Only adds boundary snapshots; all transitions remain in the lesson Engine."""
    def __init__(self, requests):
        super().__init__(requests)
        self.snapshots = []

    def snapshot(self, boundary):
        row = {
            'tick': self.ticks,
            'boundary': boundary,
            'computed': {r.request_id: r.computed for r in self.requests},
            'output': {r.request_id: r.output for r in self.requests},
            'pending': [(r.request_id, text) for r, text in self.pending],
            'blocks': {r.request_id: list(r.blocks) for r in self.requests},
            'live_refs': sum(self.pool.refs),
            'finished': sorted(self.finished),
        }
        self.snapshots.append(row)
        return row

    def prefill(self):
        super().prefill()
        self.snapshot('after-prefill')

    def decode(self):
        super().decode()
        self.snapshot('after-decode')

    def flush(self):
        self.snapshot('before-flush')
        super().flush()
        self.snapshot('after-flush')


def run_case():
    engine = ObservedEngine([
        lesson.Request('A', 7, 'true'),
        lesson.Request('B', 3, 'false'),
        lesson.Request('C', 5, 'true'),
    ])
    result = engine.run()
    before = next(row for row in engine.snapshots if row['boundary'] == 'before-flush')
    after = engine.snapshots[-1]
    assert before['output'] == {'A': 'true', 'B': 'false', 'C': 'tr'}
    assert before['pending'] == [('C', 'ue')]
    assert before['blocks']['C'] == [6, 7, 8]
    assert before['live_refs'] == 3
    assert result == ['true', 'false', 'true']
    assert after['pending'] == [] and after['live_refs'] == 0
    emitted = [event for event in engine.trace if event[0] == 'emit']
    engine.flush()  # A repeated drain must not emit the tail twice.
    assert [event for event in engine.trace if event[0] == 'emit'] == emitted
    return engine, result


if __name__ == '__main__':
    engine, result = run_case()
    for row in engine.snapshots:
        if row['tick'] >= 7:
            print(json.dumps(row, ensure_ascii=False))
    print('result:', result)
    print('checks: tail delivered once; pending empty; live refs zero')
