"""Executable integrated CPU acceptance. No upstream imports or device claims."""
from observe_lesson import run_case, load

def check():
    results = []
    for stage in range(1, 10):
        engine, output, snapshots = run_case(stage)
        assert output == ['true', 'false', 'true'], (stage, output)
        assert engine.finished == {'A', 'B', 'C'}
        assert not engine.waiting and not engine.running
        if stage >= 3:
            assert sum(engine.pool.refs) == 0
            assert all(not request.blocks for request in engine.requests)
        if stage >= 6:
            assert not engine.pending
        results.append((stage, engine.ticks))
    report = engine.report()
    assert report['verification_slots'] == 14
    assert report['committed_characters'] == 13
    assert report['logical_ticks'] == 15
    pressure, output, states = run_case(3, blocks=8)
    assert output == ['true', 'false', 'true']
    assert any('C' in row['waiting'] and row['live_refs'] == 5 for row in states)
    assert sum(pressure.pool.refs) == 0
    lesson = load(8)
    broken = lesson.Engine([lesson.Request('bad', 3, 'tx')])
    try:
        broken.run()
    except ValueError:
        assert sum(broken.pool.refs) == 0
        assert not broken.pending and not broken.waiting and not broken.running
    else:
        raise AssertionError('invalid oracle must fail')
    return {'stages_and_ticks': results, 'final_report': report,
            'pressure_and_failure_cleanup': 'passed', 'scope': 'CPU teaching simulation only'}

if __name__ == '__main__':
    import json
    print(json.dumps(check(), ensure_ascii=False, indent=2))
