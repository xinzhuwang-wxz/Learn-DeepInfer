# 03 · 内存不是一个计数器

> 给同一引擎加一池有限的块。请求要先取得所有权才能处理输入，最后一次提交后才归还；等待容量时，系统还必须允许持有资源的请求继续完成。

## 让C等在门口，再问谁能把门打开

A/B/C仍分别需要7、3、5个prompt位置，答案仍是`true / false / true`。本课为每请求一次性预留`ceil((prompt_length + answer_length) / 4)`块，所以A需3块、B需2块、C需3块。

默认池有10个编号，其中0保留，实际可用9块，三个请求的8块都能容纳。为了让同一输入产生真实背压，仅将池改成8个编号，即7个可用块：

```bash
python lessons/03_block_lifecycle/code.py
python examples/observe_lesson.py 3 --blocks 8
```

A、B拿走5块后只剩2块，C需要3块，必须等待。若引擎始终坚持“waiting非空就prefill”，A、B不会继续生成，也永远不会释放块；C等的资源无法出现。

![C暂时缺一块时必须推进A/B，A完成后C复用块1到3](../assets/block-ownership.svg)

[放大查看背压与块所有权图](../assets/block-ownership.svg)

块只是可复用编号和引用计数，没有任何KV tensor。上面的预留公式知道完整oracle答案，是教学假设，不是DeepInfer的实际显存公式。

## 1. 分清OK、LATER和NEVER

来源：[Blocks.status，16–22行](../lessons/03_block_lifecycle/code.py#L16-L22)。

<!-- code: lessons/03_block_lifecycle/code.py:L16-L22 -->
```python
def status(self, count):
    if count < 0:
        raise ValueError('negative allocation')
    if count >= len(self.refs):
        return 'NEVER'
    free = sum(n == 0 for n in self.refs[1:])
    return 'OK' if free >= count else 'LATER'
```

对8个编号的池：

- 总需求3块、当前有2空闲：LATER。别的请求结束可以改变结论。
- 总需求8块：NEVER。最多只有7可用，再等也无法满足。
- 总需求3块、当前有5空闲：OK。

`count >= len(refs)`而不是`count > len(refs)`，是因为0号块不参与分配。把永久不可满足当作暂时等待会造成无意义循环；把暂时不足当成错误则会丢掉本可完成的请求。

实际分配也必须先检查状态。来源：[allocate，24–30行](../lessons/03_block_lifecycle/code.py#L24-L30)。

<!-- code: lessons/03_block_lifecycle/code.py:L24-L30 -->
```python
def allocate(self, count):
    if self.status(count) != 'OK':
        raise ValueError(self.status(count))
    ids = [i for i in range(1, len(self.refs)) if self.refs[i] == 0][:count]
    for i in ids:
        self.refs[i] = 1
    return ids
```

只从编号1开始选引用为0的块，给新所有者设为1，返回完整block列表。这样请求可以精确归还自己持有的块，而不是在结束时笼统执行“空闲数加3”。

## 2. 准入决定发生在输入进度之前

来源：[prefill的分配段，81–89行](../lessons/03_block_lifecycle/code.py#L81-L89)。

<!-- code: lessons/03_block_lifecycle/code.py:L81-L89 -->
```python
request = self.waiting[0]
if not request.blocks:
    count = (request.prompt_length + len(request.answer) + 3)//4
    status = self.pool.status(count)
    if status == 'NEVER':
        raise ValueError('request exceeds toy block capacity')
    if status == 'LATER':
        break
    request.blocks = self.pool.allocate(count)
```

只有尚未持有块时才分配，避免每个chunk重复占用一遍。NEVER显式报错；LATER结束本轮准入，不修改C的computed。成功后才继续上一课的take计算。

C等待时，A/B拥有的块不能“为了让C开始”直接归还：它们的请求还活着。资源规划必须服从所有权，而不是只追求waiting队列变短。

## 3. 背压不是停机：让能释放资源的工作推进

来源：[主循环，124–130行](../lessons/03_block_lifecycle/code.py#L124-L130)。

<!-- code: lessons/03_block_lifecycle/code.py:L124-L130 -->
```python
if self.waiting:
    before = len(self.trace)
    self.prefill()
    if len(self.trace) == before:
        self.decode()  # capacity wait must not freeze running requests
else:
    self.decode()
```

这里用trace长度变化作为教学版的进度信号。prefill没有新增事件，就转去decode。如果本轮B刚处理完prompt，trace已增加，即使随后C被挡住，也会等下一轮才decode。真实服务更适合显式返回调度结果，不应依赖日志多少判断业务进度。

`emit()`在请求完成时释放，而不是每输出一个字符就释放。来源：[emit，68–76行](../lessons/03_block_lifecycle/code.py#L68-L76)。

<!-- code: lessons/03_block_lifecycle/code.py:L68-L76 -->
```python
def emit(self, request, text):
    request.output += text
    self.trace.append(('emit', request.request_id, text))
    if len(request.output) == len(request.answer):
        self.finished.add(request.request_id)
        self.running = [r for r in self.running if r is not request]
        for block in request.blocks:
            self.pool.free(block)
        request.blocks.clear()
```

释放后清空`request.blocks`，避免后续清理再次归还同一份所有权。第06课把提交延迟后，这个时刻也会后移；不能看见“采样完成”就先解除全部引用。

## 4. A/B/C的容量压力走读

以下来自`observe_lesson.py 3 --blocks 8`。同一tick可能先尝试prefill、再执行decode，表中选这一tick最后一个有效动作。

| tick | 动作 | A/B/C输出 | 块所有者 | 活引用和 |
|---:|---|---|---|---:|
| 2 | A完成prompt，B处理1 | ∅ / ∅ / ∅ | A:1–3，B:4–5 | 5 |
| 3 | B完成prompt，C等容量 | ∅ / ∅ / ∅ | 同上 | 5 |
| 4 | C无进度，转decode | t / f / ∅ | 同上 | 5 |
| 7 | A完成，释放3块 | true / fals / ∅ | B:4–5 | 2 |
| 8 | C得到已释放的1–3，处理4 | true / fals / ∅ | B:4–5，C:1–3 | 5 |
| 9 | C完成prompt | true / fals / ∅ | 同上 | 5 |
| 10 | B完成，C开始输出 | true / false / t | C:1–3 | 3 |
| 13 | C完成 | true / false / true | 空 | 0 |

这里的复用不是跨请求复用旧KV内容。C拿到A用过的**编号**，表示槽位可重新使用；教学模型不保存KV值，不能证明“命中了A的prefix cache”。

## 5. 最后一个所有者才决定可复用

现在主线每块只有一个请求所有者，但块管理器还支持pin。独立探针让同一块有两个引用：

```bash
python examples/s03_block_lifecycle.py
```

应先看到`[0, 1, 0, 0]`，再看到`[0, 0, 0, 0]`。来源：[pin/free与活块检查，32–42行](../lessons/03_block_lifecycle/code.py#L32-L42)。

<!-- code: lessons/03_block_lifecycle/code.py:L32-L42 -->
```python
def pin(self, i):
    self._live(i)
    self.refs[i] += 1

def free(self, i):
    self._live(i)
    self.refs[i] -= 1

def _live(self, i):
    if not 0 < i < len(self.refs) or self.refs[i] <= 0:
        raise ValueError('not a live block')
```

分配产生引用1；pin变成2；第一位释放变成1，此时allocate不能选它；最后一位释放才变0。重复free必须报错，不能让计数变负。

DeepInfer对应的C++实现还维护空闲链表。下面是归还边界：只有ref_count归零才把块放回free list。来源：[block_manager.hpp，95–103行](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/block_manager.hpp#L95-L103)。

<!-- code: reference/DeepInfer/csrc/engine/block_manager.hpp:L95-L103 -->
```cpp
void free(int block_id) {
    TORCH_CHECK(block_id > 0 && block_id < static_cast<int>(_blocks.size()));
    auto &block = _blocks[block_id];
    block.ref_count -= 1;
    TORCH_CHECK(block.ref_count >= 0, "ref_count corrupted: ref_count < 0");
    if (block.ref_count == 0) {
        _push_back(block_id);
    }
}
```

外层BlockManager提供互斥和批量pin/free；`can_allocate_nolock()`使用同样的三态区分。真实引用与空闲链表应同时一致，只有一个free计数无法表达“哪一块仍被另一条请求使用”。[块管理完整范围](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/block_manager.hpp#L27-L178)

真实调度还要考虑state容量、fork时pin的父块、尾块clone等。本课不模拟这些分支，也不把“引用归零”说成释放了底层GPU缓存张量。[准入与fork处理](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/scheduler.hpp#L372-L438)

## 6. 错误路径同样有资源账

来源：[run的失败清理，106–117行](../lessons/03_block_lifecycle/code.py#L106-L117)。

<!-- code: lessons/03_block_lifecycle/code.py:L106-L117 -->
```python
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
```

教学策略是整批失败：遍历所有仍持有的块归还、清空队列，然后重新抛出异常。它不吞掉失败，也没有实现真实服务的逐请求隔离。只要一次正常输出成功还不够，还要验证失败后的引用归零。

公开DeepInfer具有设备KV/state分配机制，但完整跨机器缓存服务、外部编排属于另一个层级。`save_kv_cache`等扩展入口与真正实现存储服务不是同一回事。[扩展钩子](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/engine/llm_engine.py#L800-L829) [官方发布边界](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/README.md#L1-L96)

## 7. 故障与可观察练习

1. **取消背压回退。** 在副本中删除“prefill无进度就decode”的分支，仍用A/B/C和总块数8。预期C持续waiting，A/B不再增长，最后触发活性保护。定位第一个不再前进的tick，而不只看异常文字。
2. **漏掉最终释放。** 删除emit中的free。最终内容可能仍对，但`sum(pool.refs)`不再为0；更大队列还会被泄漏的所有权挡住。
3. **提前释放。** 如果每次emit都free，未结束请求将失去块；继续free会报错，若错误地清空列表则可能不报错却失去所有权。断言“活请求需要的每个块都仍有引用”，不能只期待异常。
4. **共享引用。** 分配后pin一次，第一次free后断言该ID不会被再次分配；第二次free后才可复用。再free一次必须失败。
5. **不可满足。** 让单请求需要超过全部可用块，断言NEVER分支立即失败、waiting清空、已分配引用归零，而不是靠1000步警报退出。

### 参考答案：哪一项先坏？

取消背压回退首先破坏活性，C始终waiting且A/B不再推进；漏free首先破坏终态引用账；每次emit就free首先破坏活请求所有权。共享块经allocate→pin后的引用是2，第一次free后为1，第二次为0，第三次应失败。NEVER不是等待信号：单请求需求超过7个可用块时，再多decode也不能让它装入总块数8的池，应该明确拒绝并走失败清理。

[完整代码](../lessons/03_block_lifecycle/code.py) · [相邻差异](../lessons/03_block_lifecycle/delta.patch)。下一课把“我想做prefill”与“我真的能做prefill”扩展成多个参与者必须一致的阶段选择。

