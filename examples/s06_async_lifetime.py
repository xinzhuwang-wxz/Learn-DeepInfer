"""Logical event model, not concurrency/performance benchmark or GPU execution."""
from copy import deepcopy

class Pipeline:
    def __init__(self):
        self.pending = None
        self.events = []

    def step(self, token):
        self.events.append(('forward', token))
        emitted = [] if self.pending is None else [self.pending]
        if self.pending is not None:
            self.events.append(('postprocess', self.pending))
        self.events.append(('sample', token))
        self.pending = token
        return emitted

    def flush(self):
        emitted = [] if self.pending is None else [self.pending]
        if self.pending is not None:
            self.events.append(('postprocess', self.pending))
        self.pending = None
        return emitted


def borrowed_demo():
    pool = {'token': 10}
    borrowed = pool
    owned = deepcopy(pool)
    pool['token'] = 11
    return borrowed['token'], owned['token']

if __name__ == '__main__':
    p = Pipeline()
    print('visible after step:', p.step(10), p.step(11))
    print('tail only appears after flush:', p.flush())
    print('borrowed vs copied:', borrowed_demo())
