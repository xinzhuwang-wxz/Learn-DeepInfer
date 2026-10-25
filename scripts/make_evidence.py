#!/usr/bin/env python3
from pathlib import Path
import json,importlib.util,sys,difflib,html
ROOT=Path(__file__).resolve().parents[1]
(ROOT/'reports').mkdir(exist_ok=True)
lessons=sorted((ROOT/'lessons').glob('*/code.py'))
previous=''
rows=[]
for p in lessons:
    code=p.read_text();diff=''.join(difflib.unified_diff(previous.splitlines(True),code.splitlines(True),fromfile='previous/code.py',tofile=p.parent.name+'/code.py'))
    (p.parent/'delta.patch').write_text(diff)
    rows.append(f'| {p.parent.name} | {len(code.splitlines())} | {sum(x.startswith("+") and not x.startswith("+++") for x in diff.splitlines())} | {sum(x.startswith("-") and not x.startswith("---") for x in diff.splitlines())} |')
    previous=code

cases=[]
for number,pressure in [(1,False),(3,True),(6,False),(9,False)]:
    p=lessons[number-1];name='trace_'+str(number)
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    requests=[m.Request('A',7,'true'),m.Request('B',3,'false'),m.Request('C',5,'true')]
    if pressure: requests=[m.Request(str(i),7,'false') for i in range(6)]
    engine=m.Engine(requests);events=[]
    def snap(event):
        events.append({'index':len(events),'tick':engine.ticks,'event':event,'requests':[{'id':r.request_id,'prompt':r.prompt_length,'computed':getattr(r,'computed',0),'output':r.output,'blocks':list(getattr(r,'blocks',[]))} for r in engine.requests],'refs':list(engine.pool.refs) if hasattr(engine,'pool') else [],'pending':[(r.request_id,t) for r,t in getattr(engine,'pending',[])]})
    class Recorder(list):
        def append(self,event):
            super().append(event);snap(event)
    engine.trace=Recorder();snap(('initial',));engine.run();snap(('complete',engine.report()))
    cases.append({'id':p.parent.name,'title':f'{number:02} '+('内存背压 · 6 条请求' if pressure else 'A/B/C 基准轨迹'),'events':events,'report':engine.report()})
(ROOT/'assets/trace-data.js').write_text('window.LEARN_TRACES = '+json.dumps(cases,ensure_ascii=False)+';\n')
(ROOT/'reports/trace-data.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n')
story=['### 静态取证表','\n快照抓取在 trace.append 的准确时刻；例如 emit 事件已更新输出，但随后执行的资源释放可能尚未发生。最后 complete 快照才是最终资源验收。\n','| 阶段 / 场景 | 逻辑 tick | 最终输出 | 最终块引用和 | pending |','|---|---:|---|---:|---:|']
for c in cases:
    e=c['events'][-1];story.append(f"| {c['title']} | {e['tick']} | {' / '.join(r['output'] for r in e['requests'])} | {sum(e['refs'])} | {len(e['pending'])} |")
story+=['\n#### 最终阶段的关键事件','\n以下为实际运行事件。运行 `python scripts/make_evidence.py` 可在 `reports/trace-data.json` 重新生成完整记录。\n','| 事件号 | tick | 动作 | 已提交输出 A / B / C | pending |','|---:|---:|---|---|---|']
final=cases[-1]
selected=[0]+[i for i,e in enumerate(final['events']) if e['event'][0] in ['prefill','sample-pending','flush','complete']]
for i in selected:
    e=final['events'][i]
    action=e['event'][0]
    if action in ('prefill','sample-pending'): action += ' ' + str(e['event'][1]) + ' ' + repr(e['event'][2])
    elif action=='flush': action += ' ' + str(e['event'][1])
    story.append(f"| {i} | {e['tick']} | {action} | {' / '.join(r['output'] or '∅' for r in e['requests'])} | {str(e['pending'])} |")
source=(ROOT/'docs/09-evidence.md').read_text();start=source.find('[[TRACE_STORYBOARD]]')
if start>=0:source=source.replace('[[TRACE_STORYBOARD]]','\n'.join(story))
else:source=source[:source.index('### 静态取证表')]+'\n'.join(story)+'\n'
(ROOT/'docs/09-evidence.md').write_text(source)
codebook=['# 12 · 可运行代码册与阶段差异','\n以下是九个独立完整快照，与网页下载的 code.py 同源。所有源码均为原创教学实现，标准库、无联网、无模型。每个快照可单独保存成 code.py 运行。\n','## 每课到底增加了多少','\n行数是实际文件计数；新增/删除按相邻完整快照 unified diff 统计，包含说明与格式变化，不代表算法复杂度。\n','| 阶段 | 行数 | 新增行 | 删除行 |','|---|---:|---:|---:|',*rows]
for p in lessons:
    codebook.extend(['\n## '+p.parent.name, '\n```python\n'+p.read_text()+'```\n'])
codebook.extend(['\n## 附加实验：计算路径参考', '\n```python\n'+(ROOT/'examples/s10_tensor_path.py').read_text()+'```\n'])
(ROOT/'docs/12-code.md').write_text('\n'.join(codebook))
print('Generated diffs, 4 actual trace cases, static storyboard and full codebook.')
