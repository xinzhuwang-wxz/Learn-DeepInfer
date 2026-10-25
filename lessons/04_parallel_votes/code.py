"""MiniInfer stage 4: parallel_votes.
Standalone standard-library teaching simulation. No model, tensor, network or device execution.
Run: python lessons/04_parallel_votes/code.py
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
        lengths = [1]*len(active)
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
