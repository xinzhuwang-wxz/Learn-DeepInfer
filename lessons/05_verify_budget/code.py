"""MiniInfer stage 5: verify_budget.
Standalone standard-library teaching simulation. No model, tensor, network or device execution.
Run: python lessons/05_verify_budget/code.py
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field

"""Reference-counted block teaching model. Zero is reserved; no KV tensor allocation."""
class Blocks:
    def __init__(self, total):
        if total < 2:
            raise ValueError('need sentinel plus at least one usable block')
        self.refs = [0] * total

    def status(self, count):
        if count < 0:
            raise ValueError('negative allocation')
        if count >= len(self.refs):
            return 'NEVER'
        free = sum(n == 0 for n in self.refs[1:])
        return 'OK' if free >= count else 'LATER'

    def allocate(self, count):
        if self.status(count) != 'OK':
            raise ValueError(self.status(count))
        ids = [i for i in range(1, len(self.refs)) if self.refs[i] == 0][:count]
        for i in ids:
            self.refs[i] = 1
        return ids

    def pin(self, i):
        self._live(i)
        self.refs[i] += 1

    def free(self, i):
        self._live(i)
        self.refs[i] -= 1

    def _live(self, i):
        if not 0 < i < len(self.refs) or self.refs[i] <= 0:
            raise ValueError('not a live block')


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


"""Simplified MTP budget objective. Synthetic costs/rates, not DSpark or timing."""
def choose(rates, curve, sampler_ms=0.0):
    if not rates or not curve or sampler_ms < 0:
        raise ValueError('nonempty requests/curve and nonnegative sampler cost required')
    if any(not 0 <= p <= 1 for seq in rates for p in seq):
        raise ValueError('confidence must be in [0,1]')
    if any(ms <= 0 for _, ms in curve) or any(a[0] >= b[0] for a,b in zip(curve,curve[1:])):
        raise ValueError('positive cost and increasing graph sizes required')
    queue = sorted(((p, i) for i, seq in enumerate(rates) for p in seq), key=lambda x:(-x[0],x[1]))
    expected = float(len(rates))
    best = None
    for extra in range(len(queue)+1):
        total = len(rates)+extra
        graph = next(((size,ms) for size,ms in curve if size >= total), None)
        if graph is not None:
            score = expected/(graph[1]+sampler_ms)
            if best is None or score > best[0]:
                best = score, extra, total, graph[0]
        if extra < len(queue):
            expected += queue[extra][0]
    if best is None:
        raise ValueError('no graph can hold golden tokens')
    lengths = [1]*len(rates)
    for _, i in queue[:best[1]]:
        lengths[i] += 1
    return {'query_lengths':lengths, 'total_q':best[2], 'graph_size':best[3], 'synthetic_score':round(best[0],4)}


@dataclass
class Request:
    request_id: str
    prompt_length: int
    answer: str
    computed: int = 0
    output: str = ''
    blocks: list[int] = field(default_factory=list)

class Engine:
    def __init__(self, requests):
        if any(r.prompt_length<1 or not r.answer for r in requests):
            raise ValueError('positive prompt length and nonempty oracle answer required')
        if len({r.request_id for r in requests}) != len(requests):
            raise ValueError('duplicate request ID')
        self.requests = requests
        self.trace = []
        self.ticks = 0
        self.finished = set()
        self.waiting = list(requests)
        self.running = []
        self.pool = Blocks(10)  # 9 usable blocks, ID 0 reserved
        self.verification_slots = 0

    def emit(self, request, text):
        request.output += text
        self.trace.append(('emit', request.request_id, text))
        if len(request.output) == len(request.answer):
            self.finished.add(request.request_id)
            self.running = [r for r in self.running if r is not request]
            for block in request.blocks:
                self.pool.free(block)
            request.blocks.clear()

    def prefill(self):
        budget = 4
        while self.waiting and budget:
            request = self.waiting[0]
            if not request.blocks:
                count = (request.prompt_length + len(request.answer) + 3)//4
                status = self.pool.status(count)
                if status == 'NEVER':
                    raise ValueError('request exceeds toy block capacity')
                if status == 'LATER':
                    break
                request.blocks = self.pool.allocate(count)
            take = min(request.prompt_length-request.computed, budget)
            request.computed += take
            budget -= take
            self.trace.append(('prefill', request.request_id, take))
            if request.computed == request.prompt_length:
                self.running.append(self.waiting.pop(0))

    def decode(self):
        active = list(self.running)
        offsets = {r.request_id:len(r.output) for r in active}
        if not active:
            return
        allocation = choose([[.9,.5] for _ in active], [(2,1.0),(4,1.3),(8,3.0)])
        lengths = allocation['query_lengths']
        self.verification_slots += sum(lengths)
        self.trace.append(('verify-plan', lengths))
        for request,count in zip(active,lengths):
            start = offsets[request.request_id]
            text = request.answer[start:start+count]  # fake oracle, NOT a model
            self.emit(request,text)

    def run(self):
        try:
            return self._run()
        except Exception:
            # Teaching policy: fail this whole batch, then release every live owner.
            for request in self.requests:
                for block in request.blocks:
                    self.pool.free(block)
                request.blocks.clear()
            self.waiting.clear()
            self.running.clear()
            raise

    def _run(self):
        while len(self.finished) < len(self.requests):
            self.ticks += 1
            if self.ticks > 1000:
                raise RuntimeError('teaching liveness guard: inspect trace')
            # A simulated peer is idle. This is a vote model, not distributed execution.
            want = bool(self.waiting)
            can = want and (bool(self.waiting[0].blocks) or self.pool.status((self.waiting[0].prompt_length+len(self.waiting[0].answer)+3)//4)=='OK')
            if want and self.pool.status((self.waiting[0].prompt_length+len(self.waiting[0].answer)+3)//4)=='NEVER':
                raise ValueError('request exceeds toy block capacity')
            chosen = stage([want,False],[can,False],self.ticks,2,3)
            self.trace.append(('phase',chosen))
            if chosen == 'prefill':
                self.prefill()
            else:
                self.decode()
        return [r.output for r in self.requests]

    def report(self):
        return {'outputs':[r.output for r in self.requests], 'logical_ticks':self.ticks,
                'committed_characters':sum(len(r.output) for r in self.requests),
                'measurement':'logical simulation only; no tokens/s or device timing'}

if __name__ == '__main__':
    engine = Engine([Request('A',7,'true'), Request('B',3,'false'), Request('C',5,'true')])
    engine.run()
    print(json.dumps(engine.report(),ensure_ascii=False,indent=2))
    print('trace:')
    for event in engine.trace:
        print(event)
