"""MiniInfer stage 2: chunk_budget.
Standalone standard-library teaching simulation. No model, tensor, network or device execution.
Run: python lessons/02_chunk_budget/code.py
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field

@dataclass
class Request:
    request_id: str
    prompt_length: int
    answer: str
    computed: int = 0
    output: str = ''

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

    def emit(self, request, text):
        request.output += text
        self.trace.append(('emit', request.request_id, text))
        if len(request.output) == len(request.answer):
            self.finished.add(request.request_id)
            self.running = [r for r in self.running if r is not request]

    def prefill(self):
        budget = 4
        while self.waiting and budget:
            request = self.waiting[0]
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
        while len(self.finished) < len(self.requests):
            self.ticks += 1
            if self.ticks > 1000:
                raise RuntimeError('teaching liveness guard: inspect trace')
            if self.waiting:
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
