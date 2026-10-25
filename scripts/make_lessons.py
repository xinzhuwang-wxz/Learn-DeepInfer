#!/usr/bin/env python3
"""Generate checked-in cumulative, standalone CPU snapshots from explicit teaching features."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
FEATURES=['request_loop','chunk_budget','block_lifecycle','parallel_votes','verify_budget','async_lifetime','bounded_replay','grammar_rollback','metrics']

def probe(name):
    s=(ROOT/'examples'/name).read_text()
    return s.split("if __name__ == '__main__':")[0]

for stage,name in enumerate(FEATURES,1):
    prefix='''"""MiniInfer stage STAGE: NAME.
Standalone standard-library teaching simulation. No model, tensor, network or device execution.
Run: python lessons/STAGEPAD_NAME/code.py
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
'''.replace('STAGEPAD',f'{stage:02}').replace('STAGE',str(stage)).replace('NAME',name)
    for threshold,filename in [(3,'s03_block_lifecycle.py'),(4,'s04_parallel_votes.py'),(5,'s05_verify_budget.py'),(7,'s07_bounded_replay.py'),(8,'s08_grammar_rollback.py')]:
        if stage>=threshold:prefix+='\n'+probe(filename)
    prefix+='''
@dataclass
class Request:
    request_id: str
    prompt_length: int
    answer: str
    computed: int = 0
    output: str = ''
    blocks: list[int] = field(default_factory=list)
    matcher: object = None

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
'''
    prefix+=('        self.waiting = list(requests)\n        self.running = []\n' if stage>=2 else '        self.waiting = []\n        self.running = list(requests)\n')
    if stage>=3:prefix+='        self.pool = Blocks(10)  # 9 usable blocks, ID 0 reserved\n'
    if stage>=5:prefix+='        self.verification_slots = 0\n'
    if stage>=6:prefix+='        self.pending = []\n'
    if stage>=8:prefix+='        for request in requests:\n            request.matcher = Matcher()\n'
    prefix+='''
    def emit(self, request, text):
'''
    if stage>=8:prefix+='''        for token in text:
            if not request.matcher.accept(token):
                raise ValueError(f'grammar rejected {token!r} after {request.matcher.prefix!r}')
'''
    prefix+='''        request.output += text
        self.trace.append(('emit', request.request_id, text))
        if len(request.output) == len(request.answer):
            self.finished.add(request.request_id)
            self.running = [r for r in self.running if r is not request]
'''
    if stage>=3:prefix+='''            for block in request.blocks:
                self.pool.free(block)
            request.blocks.clear()
'''
    if stage>=2:
        prefix+='''
    def prefill(self):
        budget = 4
        while self.waiting and budget:
            request = self.waiting[0]
'''
        if stage>=3:prefix+='''            if not request.blocks:
                count = (request.prompt_length + len(request.answer) + 3)//4
                status = self.pool.status(count)
                if status == 'NEVER':
                    raise ValueError('request exceeds toy block capacity')
                if status == 'LATER':
                    break
                request.blocks = self.pool.allocate(count)
'''
        prefix+='''            take = min(request.prompt_length-request.computed, budget)
            request.computed += take
            budget -= take
            self.trace.append(('prefill', request.request_id, take))
'''
        if stage>=7:prefix+='''            keys = visible(request.computed-1, request.prompt_length, 4, 6, 2)
            self.trace.append(('swa-visible', request.request_id, keys))
'''
        prefix+='''            if request.computed == request.prompt_length:
                self.running.append(self.waiting.pop(0))

'''
    prefix+='''    def decode(self):
        active = list(self.running)
'''
    if stage>=6:
        prefix+='''        # Snapshot positions include one in-flight batch; never resubmit its tokens.
        offsets = {r.request_id:len(r.output)+sum(len(t) for q,t in self.pending if q is r) for r in active}
        active = [r for r in active if offsets[r.request_id] < len(r.answer)]
        self.trace.append(('forward', [r.request_id for r in active]))
        previous, self.pending = self.pending, []
        for request,text in previous:
            self.emit(request,text)
'''
    else:prefix+='        offsets = {r.request_id:len(r.output) for r in active}\n'
    if stage>=5:
        prefix+='''        if not active:
            return
        allocation = choose([[.9,.5] for _ in active], [(2,1.0),(4,1.3),(8,3.0)])
        lengths = allocation['query_lengths']
        self.verification_slots += sum(lengths)
        self.trace.append(('verify-plan', lengths))
'''
    else:prefix+='        lengths = [1]*len(active)\n'
    prefix+='''        for request,count in zip(active,lengths):
            start = offsets[request.request_id]
            text = request.answer[start:start+count]  # fake oracle, NOT a model
'''
    if stage>=6:prefix+='''            if text:
                self.pending.append((request,text))
                self.trace.append(('sample-pending',request.request_id,text))
'''
    else:prefix+='            self.emit(request,text)\n'
    if stage>=6:
        prefix+='''
    def flush(self):
        previous, self.pending = self.pending, []
        self.trace.append(('flush',len(previous)))
        for request,text in previous:
            self.emit(request,text)
'''
    prefix+='''
    def run(self):
        while len(self.finished) < len(self.requests):
            self.ticks += 1
            if self.ticks > 1000:
                raise RuntimeError('teaching liveness guard: inspect trace')
'''
    if stage>=6:
        prefix+='''            if not self.waiting and all(len(r.output)+sum(len(t) for q,t in self.pending if q is r)>=len(r.answer) for r in self.running):
                self.flush()
                continue
'''
    if stage>=4:
        prefix+='''            # A simulated peer is idle. This is a vote model, not distributed execution.
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
'''
    elif stage>=2:
        if stage>=3:
            prefix+='''            if self.waiting:
                before = len(self.trace)
                self.prefill()
                if len(self.trace) == before:
                    self.decode()  # capacity wait must not freeze running requests
            else:
                self.decode()
'''
        else:prefix+='''            if self.waiting:
                self.prefill()
            else:
                self.decode()
'''
    else:prefix+='            self.decode()\n'
    prefix+='''        return [r.output for r in self.requests]

    def report(self):
        return {'outputs':[r.output for r in self.requests], 'logical_ticks':self.ticks,
'''
    prefix+=('                \'verification_slots\':self.verification_slots,\n' if stage>=9 else '')
    prefix+='''                'committed_characters':sum(len(r.output) for r in self.requests),
                'measurement':'logical simulation only; no tokens/s or device timing'}

if __name__ == '__main__':
    engine = Engine([Request('A',7,'true'), Request('B',3,'false'), Request('C',5,'true')])
    engine.run()
    print(json.dumps(engine.report(),ensure_ascii=False,indent=2))
    print('trace:')
    for event in engine.trace:
        print(event)
'''
    if stage >= 3:
        cleanup = """    def run(self):
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
"""
        if stage >= 6: cleanup += "            self.pending.clear()\n"
        cleanup += "            raise\n\n    def _run(self):"
        prefix = prefix.replace("    def run(self):", cleanup)
    if stage < 2: prefix = prefix.replace('    computed: int = 0\n', '')
    if stage < 3: prefix = prefix.replace('    blocks: list[int] = field(default_factory=list)\n', '')
    if stage < 8: prefix = prefix.replace('    matcher: object = None\n', '')
    out=ROOT/'lessons'/f'{stage:02}_{name}';out.mkdir(exist_ok=True,parents=True)
    (out/'code.py').write_text(prefix)
print('Generated 9 standalone cumulative snapshots.')
