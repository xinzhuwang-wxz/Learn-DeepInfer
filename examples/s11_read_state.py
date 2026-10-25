"""Observe a shared block's ownership without loading upstream dependencies."""
from observe_lesson import load

def walk():
    lesson = load(3)
    pool = lesson.Blocks(3)  # 0 reserved; 1 and 2 usable
    block = pool.allocate(1)[0]
    states = [('allocate', pool.refs.copy(), pool.status(2))]
    pool.pin(block)
    states.append(('pin', pool.refs.copy(), pool.status(2)))
    pool.free(block)
    states.append(('first free', pool.refs.copy(), pool.status(2)))
    assert pool.refs[block] == 1 and pool.status(2) == 'LATER'
    pool.free(block)
    states.append(('last free', pool.refs.copy(), pool.status(2)))
    assert pool.refs[block] == 0 and pool.status(2) == 'OK'
    try:
        pool.free(block)
    except ValueError as error:
        states.append(('double free', str(error), pool.refs.copy()))
    else:
        raise AssertionError('double free should be rejected')
    return states

if __name__ == '__main__':
    import json
    print(json.dumps(walk(), ensure_ascii=False, indent=2))
