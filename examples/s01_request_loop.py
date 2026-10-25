"""Teaching model only: request identity/order, NOT model inference. Stdlib-only."""
from dataclasses import dataclass, field

@dataclass
class Request:
    request_id: int
    target: int
    output: list[int] = field(default_factory=list)

    @property
    def finished(self):
        return len(self.output) >= self.target


def run(targets):
    if any(n < 1 for n in targets):
        raise ValueError('target lengths must be positive')
    requests = [Request(i, n) for i, n in enumerate(targets)]
    completion_order = []
    while any(not r.finished for r in requests):
        for request in requests:
            if request.finished:
                continue
            request.output.append(100 + len(request.output))  # fake token
            if request.finished:
                completion_order.append(request.request_id)
    return [r.output for r in requests], completion_order

if __name__ == '__main__':
    outputs, order = run([3, 1, 2])
    print('completion order:', order)
    print('input-ordered outputs:', outputs)
