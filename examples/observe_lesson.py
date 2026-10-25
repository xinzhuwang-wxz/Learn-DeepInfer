"""Print method-boundary state for the same A/B/C in any teaching stage.

Only imports the local standard-library lessons. No model, device or network.
Example: python examples/observe_lesson.py 3 --blocks 8
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def load(stage):
    paths = sorted((ROOT / 'lessons').glob(f'{stage:02}_*/code.py'))
    if len(paths) != 1:
        raise ValueError('stage must select exactly one local lesson')
    name = f'observed_mini_infer_{stage}'
    spec = importlib.util.spec_from_file_location(name, paths[0])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_case(stage, blocks=None):
    lesson = load(stage)
    engine = lesson.Engine([lesson.Request('A', 7, 'true'),
                            lesson.Request('B', 3, 'false'),
                            lesson.Request('C', 5, 'true')])
    if blocks is not None:
        if not hasattr(engine, 'pool'):
            raise ValueError('block capacity is introduced at stage 03')
        engine.pool = lesson.Blocks(blocks)
    snapshots = []

    def snapshot(boundary):
        snapshots.append({
            'tick': engine.ticks, 'boundary': boundary,
            'waiting': [r.request_id for r in engine.waiting],
            'running': [r.request_id for r in engine.running],
            'finished': sorted(engine.finished),
            'computed': [getattr(r, 'computed', 0) for r in engine.requests],
            'output': [r.output for r in engine.requests],
            'pending': [(r.request_id, text) for r, text in getattr(engine, 'pending', [])],
            'blocks': [list(getattr(r, 'blocks', [])) for r in engine.requests],
            'live_refs': sum(engine.pool.refs) if hasattr(engine, 'pool') else 0,
            'verification_slots': getattr(engine, 'verification_slots', 0),
        })

    for name in ('prefill', 'decode', 'flush'):
        original = getattr(engine, name, None)
        if original is None:
            continue
        def observed(original=original, name=name):
            result = original()
            snapshot('after-' + name)
            return result
        setattr(engine, name, observed)
    snapshot('initial')
    result = engine.run()
    snapshot('complete')
    return engine, result, snapshots


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', type=int, choices=range(1, 10))
    parser.add_argument('--blocks', type=int, help='total toy blocks, including reserved 0')
    args = parser.parse_args()
    engine, result, snapshots = run_case(args.stage, args.blocks)
    for row in snapshots:
        print(json.dumps(row, ensure_ascii=False))
    print('result:', result)
    print('report:', json.dumps(engine.report(), ensure_ascii=False))
