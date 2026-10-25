"""FIFO token-budget teaching model; omits CP, KV blocks, multimodal and DP voting."""
from dataclasses import dataclass

@dataclass
class Prompt:
    request_id: str
    length: int
    computed: int = 0


def step(waiting, budget):
    if budget < 1:
        raise ValueError('budget must be positive')
    selected = []
    for request in waiting:
        if request.length < 0 or not 0 <= request.computed <= request.length:
            raise ValueError('invalid prompt state')
        take = min(request.length - request.computed, budget)
        if take:
            selected.append((request.request_id, take))
            request.computed += take
            budget -= take
        if budget == 0:
            break
    return selected

if __name__ == '__main__':
    waiting = [Prompt('A', 7), Prompt('B', 3)]
    while any(p.computed < p.length for p in waiting):
        print(step(waiting, 4))
