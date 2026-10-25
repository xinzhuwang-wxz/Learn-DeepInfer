"""Observe grammar state in the local CPU teaching engine; no upstream imports."""
from __future__ import annotations
import json
from observe_lesson import load


def observe():
    lesson = load(8)
    engine = lesson.Engine([lesson.Request('A', 7, 'true'), lesson.Request('B', 3, 'false'), lesson.Request('C', 5, 'true')])
    rows = []
    for name in ('prefill', 'decode', 'flush'):
        original = getattr(engine, name)
        def wrapped(original=original, name=name):
            original()
            rows.append({'tick': engine.ticks, 'after': name,
                         'outputs': [r.output for r in engine.requests],
                         'prefixes': [r.matcher.prefix for r in engine.requests],
                         'pending': [(r.request_id, t) for r, t in engine.pending],
                         'live_refs': sum(engine.pool.refs)})
        setattr(engine, name, wrapped)
    engine.run()
    return engine, rows


def rollback_case():
    matcher = load(8).Matcher()
    rows = []
    for token in 'tru':
        accepted = matcher.accept(token)
        rows.append((token, accepted, matcher.prefix, sorted(matcher.allowed())))
    matcher.rollback(2)
    rows.append(('rollback(2)', None, matcher.prefix, sorted(matcher.allowed())))
    return rows


if __name__ == '__main__':
    for row in rollback_case():
        print(json.dumps(row, ensure_ascii=False))
    engine, rows = observe()
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    print(json.dumps(engine.report(), ensure_ascii=False))
