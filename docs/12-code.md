# 12 · 可运行代码册与阶段差异

以下是九个独立完整快照，与网页下载的 code.py 同源。所有源码均为原创教学实现，标准库、无联网、无模型。每个快照可单独保存成 code.py 运行。

## 每课到底增加了多少

行数是实际文件计数；新增/删除按相邻完整快照 unified diff 统计，包含说明与格式变化，不代表算法复杂度。

| 阶段 | 行数 | 新增行 | 删除行 |
|---|---:|---:|---:|
| 01_request_loop | 63 | 63 | 0 |
| 02_chunk_budget | 79 | 21 | 5 |
| 03_block_lifecycle | 144 | 67 | 2 |
| 04_parallel_votes | 163 | 25 | 6 |
| 05_verify_budget | 197 | 37 | 3 |
| 06_async_lifetime | 216 | 22 | 3 |
| 07_bounded_replay | 235 | 21 | 2 |
| 08_grammar_rollback | 267 | 34 | 2 |
| 09_metrics | 268 | 3 | 2 |

## 01_request_loop

```python
"""MiniInfer stage 1: request_loop.
Standalone standard-library teaching simulation. No model, tensor, network or device execution.
Run: python lessons/01_request_loop/code.py
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field

@dataclass
class Request:
    request_id: str
    prompt_length: int
    answer: str
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
        self.waiting = []
        self.running = list(requests)

    def emit(self, request, text):
        request.output += text
        self.trace.append(('emit', request.request_id, text))
        if len(request.output) == len(request.answer):
            self.finished.add(request.request_id)
            self.running = [r for r in self.running if r is not request]
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
```


## 02_chunk_budget

```python
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
```


## 03_block_lifecycle

```python
"""MiniInfer stage 3: block_lifecycle.
Standalone standard-library teaching simulation. No model, tensor, network or device execution.
Run: python lessons/03_block_lifecycle/code.py
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
            if self.waiting:
                before = len(self.trace)
                self.prefill()
                if len(self.trace) == before:
                    self.decode()  # capacity wait must not freeze running requests
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
```


## 04_parallel_votes

```python
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
```


## 05_verify_budget

```python
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
```


## 06_async_lifetime

```python
"""MiniInfer stage 6: async_lifetime.
Standalone standard-library teaching simulation. No model, tensor, network or device execution.
Run: python lessons/06_async_lifetime/code.py
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
        self.pending = []

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
        # Snapshot positions include one in-flight batch; never resubmit its tokens.
        offsets = {r.request_id:len(r.output)+sum(len(t) for q,t in self.pending if q is r) for r in active}
        active = [r for r in active if offsets[r.request_id] < len(r.answer)]
        self.trace.append(('forward', [r.request_id for r in active]))
        previous, self.pending = self.pending, []
        for request,text in previous:
            self.emit(request,text)
        if not active:
            return
        allocation = choose([[.9,.5] for _ in active], [(2,1.0),(4,1.3),(8,3.0)])
        lengths = allocation['query_lengths']
        self.verification_slots += sum(lengths)
        self.trace.append(('verify-plan', lengths))
        for request,count in zip(active,lengths):
            start = offsets[request.request_id]
            text = request.answer[start:start+count]  # fake oracle, NOT a model
            if text:
                self.pending.append((request,text))
                self.trace.append(('sample-pending',request.request_id,text))

    def flush(self):
        previous, self.pending = self.pending, []
        self.trace.append(('flush',len(previous)))
        for request,text in previous:
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
            self.pending.clear()
            raise

    def _run(self):
        while len(self.finished) < len(self.requests):
            self.ticks += 1
            if self.ticks > 1000:
                raise RuntimeError('teaching liveness guard: inspect trace')
            if not self.waiting and all(len(r.output)+sum(len(t) for q,t in self.pending if q is r)>=len(r.answer) for r in self.running):
                self.flush()
                continue
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
```


## 07_bounded_replay

```python
"""MiniInfer stage 7: bounded_replay.
Standalone standard-library teaching simulation. No model, tensor, network or device execution.
Run: python lessons/07_bounded_replay/code.py
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


"""One-dimensional SWA visibility toy. Not complete YoCo/CP/cache equivalence."""
def visible(pos, data_len, window, keep, backtrack=0, valid_start=0):
    if window < 1 or backtrack < 0 or keep < window + backtrack:
        raise ValueError('keep must cover window + backtracking')
    if not 0 <= pos < data_len or valid_start < 0:
        raise ValueError('invalid absolute position')
    lo = min(pos+1, max(0, pos-window+1, data_len-keep, valid_start))
    return tuple(range(lo, pos+1))


def compare(length=12, window=4, keep=6, backtrack=2):
    full = {p: visible(p,length,window,keep,backtrack) for p in range(length)}
    compact = {p: visible(p,length,window,keep,backtrack) for p in range(max(0,length-keep),length)}
    assert all(full[p] == keys for p,keys in compact.items())
    return compact


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
        self.pending = []

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
            keys = visible(request.computed-1, request.prompt_length, 4, 6, 2)
            self.trace.append(('swa-visible', request.request_id, keys))
            if request.computed == request.prompt_length:
                self.running.append(self.waiting.pop(0))

    def decode(self):
        active = list(self.running)
        # Snapshot positions include one in-flight batch; never resubmit its tokens.
        offsets = {r.request_id:len(r.output)+sum(len(t) for q,t in self.pending if q is r) for r in active}
        active = [r for r in active if offsets[r.request_id] < len(r.answer)]
        self.trace.append(('forward', [r.request_id for r in active]))
        previous, self.pending = self.pending, []
        for request,text in previous:
            self.emit(request,text)
        if not active:
            return
        allocation = choose([[.9,.5] for _ in active], [(2,1.0),(4,1.3),(8,3.0)])
        lengths = allocation['query_lengths']
        self.verification_slots += sum(lengths)
        self.trace.append(('verify-plan', lengths))
        for request,count in zip(active,lengths):
            start = offsets[request.request_id]
            text = request.answer[start:start+count]  # fake oracle, NOT a model
            if text:
                self.pending.append((request,text))
                self.trace.append(('sample-pending',request.request_id,text))

    def flush(self):
        previous, self.pending = self.pending, []
        self.trace.append(('flush',len(previous)))
        for request,text in previous:
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
            self.pending.clear()
            raise

    def _run(self):
        while len(self.finished) < len(self.requests):
            self.ticks += 1
            if self.ticks > 1000:
                raise RuntimeError('teaching liveness guard: inspect trace')
            if not self.waiting and all(len(r.output)+sum(len(t) for q,t in self.pending if q is r)>=len(r.answer) for r in self.running):
                self.flush()
                continue
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
```


## 08_grammar_rollback

```python
"""MiniInfer stage 8: grammar_rollback.
Standalone standard-library teaching simulation. No model, tensor, network or device execution.
Run: python lessons/08_grammar_rollback/code.py
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


"""One-dimensional SWA visibility toy. Not complete YoCo/CP/cache equivalence."""
def visible(pos, data_len, window, keep, backtrack=0, valid_start=0):
    if window < 1 or backtrack < 0 or keep < window + backtrack:
        raise ValueError('keep must cover window + backtracking')
    if not 0 <= pos < data_len or valid_start < 0:
        raise ValueError('invalid absolute position')
    lo = min(pos+1, max(0, pos-window+1, data_len-keep, valid_start))
    return tuple(range(lo, pos+1))


def compare(length=12, window=4, keep=6, backtrack=2):
    full = {p: visible(p,length,window,keep,backtrack) for p in range(length)}
    compact = {p: visible(p,length,window,keep,backtrack) for p in range(max(0,length-keep),length)}
    assert all(full[p] == keys for p,keys in compact.items())
    return compact


"""Tiny character grammar accepts exactly 'true' or 'false'. NOT xgrammar/JSON schema."""
class Matcher:
    words = ('true', 'false')
    def __init__(self):
        self.prefix = ''

    def allowed(self):
        result = {w[len(self.prefix)] for w in self.words if w.startswith(self.prefix) and len(w)>len(self.prefix)}
        if self.prefix in self.words:
            result.add('<eos>')
        return result

    def accept(self, token):
        if token not in self.allowed():
            return False
        if token != '<eos>':
            self.prefix += token
        return True

    def rollback(self, steps):
        if not 0 <= steps <= len(self.prefix):
            raise ValueError('rollback exceeds history')
        if steps:
            self.prefix = self.prefix[:-steps]


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
        self.waiting = list(requests)
        self.running = []
        self.pool = Blocks(10)  # 9 usable blocks, ID 0 reserved
        self.verification_slots = 0
        self.pending = []
        for request in requests:
            request.matcher = Matcher()

    def emit(self, request, text):
        for token in text:
            if not request.matcher.accept(token):
                raise ValueError(f'grammar rejected {token!r} after {request.matcher.prefix!r}')
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
            keys = visible(request.computed-1, request.prompt_length, 4, 6, 2)
            self.trace.append(('swa-visible', request.request_id, keys))
            if request.computed == request.prompt_length:
                self.running.append(self.waiting.pop(0))

    def decode(self):
        active = list(self.running)
        # Snapshot positions include one in-flight batch; never resubmit its tokens.
        offsets = {r.request_id:len(r.output)+sum(len(t) for q,t in self.pending if q is r) for r in active}
        active = [r for r in active if offsets[r.request_id] < len(r.answer)]
        self.trace.append(('forward', [r.request_id for r in active]))
        previous, self.pending = self.pending, []
        for request,text in previous:
            self.emit(request,text)
        if not active:
            return
        allocation = choose([[.9,.5] for _ in active], [(2,1.0),(4,1.3),(8,3.0)])
        lengths = allocation['query_lengths']
        self.verification_slots += sum(lengths)
        self.trace.append(('verify-plan', lengths))
        for request,count in zip(active,lengths):
            start = offsets[request.request_id]
            text = request.answer[start:start+count]  # fake oracle, NOT a model
            if text:
                self.pending.append((request,text))
                self.trace.append(('sample-pending',request.request_id,text))

    def flush(self):
        previous, self.pending = self.pending, []
        self.trace.append(('flush',len(previous)))
        for request,text in previous:
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
            self.pending.clear()
            raise

    def _run(self):
        while len(self.finished) < len(self.requests):
            self.ticks += 1
            if self.ticks > 1000:
                raise RuntimeError('teaching liveness guard: inspect trace')
            if not self.waiting and all(len(r.output)+sum(len(t) for q,t in self.pending if q is r)>=len(r.answer) for r in self.running):
                self.flush()
                continue
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
```


## 09_metrics

```python
"""MiniInfer stage 9: metrics.
Standalone standard-library teaching simulation. No model, tensor, network or device execution.
Run: python lessons/09_metrics/code.py
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


"""One-dimensional SWA visibility toy. Not complete YoCo/CP/cache equivalence."""
def visible(pos, data_len, window, keep, backtrack=0, valid_start=0):
    if window < 1 or backtrack < 0 or keep < window + backtrack:
        raise ValueError('keep must cover window + backtracking')
    if not 0 <= pos < data_len or valid_start < 0:
        raise ValueError('invalid absolute position')
    lo = min(pos+1, max(0, pos-window+1, data_len-keep, valid_start))
    return tuple(range(lo, pos+1))


def compare(length=12, window=4, keep=6, backtrack=2):
    full = {p: visible(p,length,window,keep,backtrack) for p in range(length)}
    compact = {p: visible(p,length,window,keep,backtrack) for p in range(max(0,length-keep),length)}
    assert all(full[p] == keys for p,keys in compact.items())
    return compact


"""Tiny character grammar accepts exactly 'true' or 'false'. NOT xgrammar/JSON schema."""
class Matcher:
    words = ('true', 'false')
    def __init__(self):
        self.prefix = ''

    def allowed(self):
        result = {w[len(self.prefix)] for w in self.words if w.startswith(self.prefix) and len(w)>len(self.prefix)}
        if self.prefix in self.words:
            result.add('<eos>')
        return result

    def accept(self, token):
        if token not in self.allowed():
            return False
        if token != '<eos>':
            self.prefix += token
        return True

    def rollback(self, steps):
        if not 0 <= steps <= len(self.prefix):
            raise ValueError('rollback exceeds history')
        if steps:
            self.prefix = self.prefix[:-steps]


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
        self.waiting = list(requests)
        self.running = []
        self.pool = Blocks(10)  # 9 usable blocks, ID 0 reserved
        self.verification_slots = 0
        self.pending = []
        for request in requests:
            request.matcher = Matcher()

    def emit(self, request, text):
        for token in text:
            if not request.matcher.accept(token):
                raise ValueError(f'grammar rejected {token!r} after {request.matcher.prefix!r}')
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
            keys = visible(request.computed-1, request.prompt_length, 4, 6, 2)
            self.trace.append(('swa-visible', request.request_id, keys))
            if request.computed == request.prompt_length:
                self.running.append(self.waiting.pop(0))

    def decode(self):
        active = list(self.running)
        # Snapshot positions include one in-flight batch; never resubmit its tokens.
        offsets = {r.request_id:len(r.output)+sum(len(t) for q,t in self.pending if q is r) for r in active}
        active = [r for r in active if offsets[r.request_id] < len(r.answer)]
        self.trace.append(('forward', [r.request_id for r in active]))
        previous, self.pending = self.pending, []
        for request,text in previous:
            self.emit(request,text)
        if not active:
            return
        allocation = choose([[.9,.5] for _ in active], [(2,1.0),(4,1.3),(8,3.0)])
        lengths = allocation['query_lengths']
        self.verification_slots += sum(lengths)
        self.trace.append(('verify-plan', lengths))
        for request,count in zip(active,lengths):
            start = offsets[request.request_id]
            text = request.answer[start:start+count]  # fake oracle, NOT a model
            if text:
                self.pending.append((request,text))
                self.trace.append(('sample-pending',request.request_id,text))

    def flush(self):
        previous, self.pending = self.pending, []
        self.trace.append(('flush',len(previous)))
        for request,text in previous:
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
            self.pending.clear()
            raise

    def _run(self):
        while len(self.finished) < len(self.requests):
            self.ticks += 1
            if self.ticks > 1000:
                raise RuntimeError('teaching liveness guard: inspect trace')
            if not self.waiting and all(len(r.output)+sum(len(t) for q,t in self.pending if q is r)>=len(r.answer) for r in self.running):
                self.flush()
                continue
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
                'verification_slots':self.verification_slots,
                'committed_characters':sum(len(r.output) for r in self.requests),
                'measurement':'logical simulation only; no tokens/s or device timing'}

if __name__ == '__main__':
    engine = Engine([Request('A',7,'true'), Request('B',3,'false'), Request('C',5,'true')])
    engine.run()
    print(json.dumps(engine.report(),ensure_ascii=False,indent=2))
    print('trace:')
    for event in engine.trace:
        print(event)
```


## 附加实验：计算路径参考

```python
"""CPU reference arithmetic: attention visibility, toy routing, exact format byte counts.
Not V4 attention, no RoPE/MLA/compressor/sink/HC, no real FP8/FP4 arithmetic or kernels.
"""
import math
import json

def attention(query, keys, values, visible):
    if not visible:
        raise ValueError('empty reference visibility')
    if len(keys) != len(values) or not query:
        raise ValueError('shape mismatch')
    if any(len(k) != len(query) for k in keys):
        raise ValueError('Q/K dimension mismatch')
    if any(len(v) != len(values[0]) for v in values):
        raise ValueError('V dimension mismatch')
    scores=[sum(a*b for a,b in zip(query,keys[i]))/math.sqrt(len(query)) for i in visible]
    peak=max(scores);weights=[math.exp(s-peak) for s in scores];denom=sum(weights)
    probs=[w/denom for w in weights]
    return [sum(p*values[i][d] for p,i in zip(probs,visible)) for d in range(len(values[0]))]

def paged_view(blocks, block_table, length):
    flat=[token for physical in block_table for token in blocks[physical]]
    if length > len(flat) or length < 0:
        raise ValueError('block table cannot cover length')
    return flat[:length]

def route(scores,bias,topk,logical_to_physical):
    """Toy: correction changes selection; raw positive scores define normalized weights.
    This is NOT a reimplementation of tile_kernels sqrtsoftplus/noaux_tc.
    """
    if len(scores)!=len(bias) or not 1<=topk<=len(scores) or any(x<0 for x in scores):
        raise ValueError('invalid toy gate inputs')
    chosen=sorted(range(len(scores)),key=lambda i:(-(scores[i]+bias[i]),i))[:topk]
    total=sum(scores[i] for i in chosen)
    if total<=0:raise ValueError('zero route weight')
    return [(i,logical_to_physical[i][0],scores[i]/total) for i in chosen]

def combine(x,routes,expert_outputs):
    """Fixed outputs stand in for grouped GEMM/SwiGLU; x is a toy shared path."""
    return [x[d]+sum(weight*expert_outputs[logical][d] for logical,_,weight in routes) for d in range(len(x))]

def kv_bytes(elements,bits,per_scale,scale_bytes):
    if elements<1 or bits not in (4,8) or per_scale<1 or scale_bytes<1 or elements*bits%8:
        raise ValueError('invalid packed format')
    return elements*bits//8 + math.ceil(elements/per_scale)*scale_bytes

if __name__=='__main__':
    keys=[[1.,0.],[0.,1.],[1.,1.],[9.,9.]]
    values=[[1.,0.],[0.,1.],[2.,2.],[99.,99.]]
    # Physical block 1 contains logical positions 0..1, physical 0 contains 2..3.
    stored=[values[2:],values[:2]]
    out=attention([1.,0.],keys,paged_view(stored,[1,0],4),[0,1])
    routes=route([.6,.3,.1],[0.,0.,.5],2,{0:[3],1:[1],2:[0,2]})
    print(json.dumps({'reference_attention':out,'logical_physical_weight':routes,
      'toy_moe_output':combine(out,routes,{0:[2.,0.],1:[0.,2.],2:[1.,1.]}),
      'format_bytes_per_token':{'V41_MXFP8':kv_bytes(512,8,32,1),'V41_FP4':kv_bytes(512,4,16,1),'FP4_INDEX':kv_bytes(128,4,32,1)},
      'scope':'reference arithmetic and layout accounting only'},indent=2))
```
