"""NORMAL-engine two-vote conceptual model. No collective or hardware is executed."""
def stage(wants, allocates, decode_step, all_interval=2, any_interval=6):
    if not wants or len(wants) != len(allocates):
        raise ValueError('rank vectors must be nonempty and aligned')
    if min(all_interval, any_interval) < 1 or decode_step < 0:
        raise ValueError('invalid interval/step')
    any_point = decode_step != 0 and decode_step % any_interval == 0
    all_point = decode_step % all_interval == 0
    if not (any_point or all_point):
        return 'decode'
    vote = any(wants) if any_point else all(wants)
    actual = any(want and can for want, can in zip(wants, allocates))
    return 'prefill' if vote and actual else 'decode'

if __name__ == '__main__':
    for s in [0, 1, 2, 6]:
        print(s, stage([True, False], [True, False], s))
    print('all want but none can allocate:', stage([True, True], [False, False], 0))
