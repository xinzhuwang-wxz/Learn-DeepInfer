"""Render the async chapter's original SVG state diagrams; no external packages."""
from pathlib import Path
from html import escape
from math import hypot
ROOT = Path(__file__).resolve().parents[1]
INK='#183b36'; MUTED='#526963'; TEAL='#087e70'; AMBER='#98651b'; RED='#a4473d'
def txt(x,y,value,size=17,weight=400,fill=INK):
    return f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}">{escape(value)}</text>'
def rect(x,y,w,h,fill='#fff',stroke='#bad2ca',rx=12):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>'
def arrow(x1,y1,x2,y2,color=TEAL):
    length=hypot(x2-x1,y2-y1); ux=(x2-x1)/length; uy=(y2-y1)/length
    bx,by=x2-9*ux,y2-9*uy
    points=f'{x2},{y2} {bx-4*uy},{by+4*ux} {bx+4*uy},{by-4*ux}'
    return f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{color}" stroke-width="2.5"/><polygon points="{points}" fill="{color}"/>'
def panel(x,y,w,h,heading,lines,accent=TEAL):
    s=rect(x,y,w,h)+rect(x+1,y+1,5,h-2,accent,accent,2)+txt(x+20,y+32,heading,20,650)
    for i,line in enumerate(lines):s+=txt(x+20,y+65+i*28,line,16)
    return s
def save(name,title,desc,width,height,body):
    svg=f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc"><title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 Z" fill="{TEAL}"/></marker></defs><g font-family="Noto Sans CJK SC, PingFang SC, Microsoft YaHei, sans-serif">{rect(0,0,width,height,'#f3f8f5','#d8e5de',18)}{body}</g></svg>'''
    (ROOT/'assets'/name).write_text(svg+'\n')

body=txt(28,42,'一个请求，三份不能混在一起的状态',25,700)
body+=txt(28,73,'第8步结束后的 A：已经产生 true，但只提交了 tr。',17,400,MUTED)
body+=panel(28,108,270,174,'已提交 · A.output',["'tr'",'len(output) = 2','emit() 才能追加'])
body+=panel(336,108,294,174,'在途 · pending',["[(A, 'ue')]",'属于 A 的在途长度 = 2','decode() 采样后保存'],AMBER)
body+=panel(668,108,264,174,'资源 · A.blocks',['[1, 2, 3]','3 个引用仍然存活','提交结束后才归还'])
body+=arrow(298,198,330,198)+txt(264,319,'下一位置 = 已提交长度 2 + 在途长度 2 = 4',20,650)
body+=txt(28,357,'下一次 decode：A 不再参与采样，但 previous 里的 A 结果仍要提交。',17,400,MUTED)
body+=panel(28,390,288,138,'① 移交旧 batch',['previous ← pending','pending ← []'])
body+=arrow(320,458,350,458)
body+=panel(358,390,268,138,'② 提交给原请求',["emit(A, 'ue')","A.output = 'true'"])
body+=arrow(630,458,660,458)
body+=panel(668,390,264,138,'③ 结束 A',['finished 加入 A','块 1、2、3 引用归零'])
body+=txt(28,570,'教学块只表示引用与复用资格；不是实际 KV tensor，也没有测量 GPU 时间。',16,400,MUTED)
save('async-state.svg','pending、输出与块引用的区别','第8步后A的output是tr，pending是A的ue，三个块仍由A持有。下一次decode跳过A采样但提交它的旧结果，然后结束并释放教学引用。',960,600,body)

body=txt(28,42,'没有下一次 forward，也有最后一次提交',25,700)
body+=txt(28,74,'A 已于第10步完成。以下完整追踪 B / C；每行右侧是该方法结束后的状态。',16,400,MUTED)
headers=[(28,'逻辑步'),(148,'本轮 forward'),(348,'提交 previous'),(580,'新的 pending'),(808,'输出 / 活引用')]
for x,label in headers:body+=txt(x,117,label,16,650)
rows=[
(138,'13 · decode',['B, C'],["B ← 'ls'"],["B: 'e'","C: 'tr'"],["B: fals","C: ∅","引用和 5"]),
(280,'14 · decode',['C','B 已无新位置'],["B ← 'e'","C ← 'tr'"],["C: 'ue'"],["B: false","C: tr","引用和 3"]),
(422,'15 · flush',['不启动'],["C ← 'ue'"],['空'],["B: false","C: true","引用和 0"])]
for y,label,forward,post,pending,status in rows:
    body+=rect(20,y,956,126,'#fff' if 'flush' not in label else '#e8f5ed')
    body+=txt(33,y+32,label,16,650)
    for x,lines in [(148,forward),(348,post),(580,pending),(808,status)]:
        for i,line in enumerate(lines):body+=txt(x,y+32+i*30,line,16)
    body+=arrow(305,y+62,331,y+62)
    if 'flush' not in label:body+=arrow(537,y+62,563,y+62)
body+=rect(28,572,940,78,'#fff5e5','#d7b77c')
body+=txt(47,602,'错误出口：在第14步后直接 return，只能返回 C = tr。',19,650,AMBER)
body+=txt(47,630,'正确出口：先提交 pending 中的 ue，再验证 pending 为空、请求完成、教学引用归零。',15,400,INK)
save('async-timeline.svg','A/B/C运行的尾部排空','第13步采样B的e和C的tr，第14步只forward C但提交B和C旧结果，第15步不forward只flush C的ue；活引用从5到3再到0。',1000,680,body)

body=txt(28,42,'输出记录与 KV 块，不是同一份所有权',25,700)
body+=txt(28,73,'DeepInfer 固定源码的生命周期边界；下方字典探针只做安全类比。',16,400,MUTED)
body+=panel(28,110,288,170,'引擎输出池',['RequestOutput 记录','raw_output 是 uint64 地址','后续 step 可释放 / 复用'])
body+=arrow(320,193,379,193)+txt(338,170,'clone',15,650,TEAL)
body+=panel(388,110,312,170,'调用者拥有的副本',['RequestOutput.clone(raw_output)','保存进 outputs[request_id]','不是只复制地址数组'])
body+=panel(736,110,236,170,'另一本账：KV块',['引用 / pin 计数','归零后可复用槽位','不是释放 GPU tensor'])
body+=rect(28,314,944,52,'#e9f3ef','#c8dbd2')
body+=txt(47,346,'flush 排空在途状态；不能由此推出所有输出池、metadata、KV 存储都已释放。',17,600)
body+=txt(28,410,'Python 别名探针：两个名字可以指向同一对象',21,650)
body+=panel(28,440,288,160,'同一个字典',["pool = {'token': 10}",'borrowed = pool',"修改 pool['token'] = 11"])
body+=arrow(320,517,379,517)
body+=panel(388,440,270,160,'borrowed 的结果',['仍指向原字典','读到 11'],AMBER)
body+=panel(696,440,276,160,'owned 的结果',['修改前 deepcopy(pool)','独立副本仍是 10'])
body+=txt(28,644,'真实 C++ 旧地址失效后不能安全读取；不保证它只是“变成下一个 token”。',16,400,RED)
save('async-ownership.svg','池中输出记录与调用者副本以及KV块的不同生命周期','RequestOutput.clone复制地址指向的记录，而不是复制地址。KV块引用是单独账本。Python字典别名实验说明共享可变对象，不能把真实失效C++指针当作安全可读的旧字典。',1000,676,body)
print('Wrote 3 async chapter SVGs.')
