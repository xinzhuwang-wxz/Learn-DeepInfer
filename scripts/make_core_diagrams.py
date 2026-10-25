"""Original mechanism diagrams. Numbers are teaching inputs, not device timings."""
from pathlib import Path
from html import escape
ROOT = Path(__file__).resolve().parents[1]
INK='#183b36'; TEAL='#087e70'; AMBER='#98651b'
def t(x,y,s,size=20,color=INK):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}">{escape(str(s))}</text>'
def box(x,y,w,h,title,lines,color=TEAL):
    z=f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="white" stroke="{color}" stroke-width="2"/>'+t(x+18,y+32,title,22,color)
    return z+''.join(t(x+18,y+65+29*i,v,18) for i,v in enumerate(lines))
def arrow(x,y,xx,yy,label=''):
    return f'<path d="M{x} {y} L{xx} {yy}" stroke="{TEAL}" stroke-width="3" marker-end="url(#a)" fill="none"/>'+ (t((x+xx)/2+8,(y+yy)/2-10,label,17) if label else '')
def write(name,title,body,foot,h=460):
    data=f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{h}" viewBox="0 0 1100 {h}" role="img" aria-label="{escape(title)}"><defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10" fill="{TEAL}"/></marker></defs><rect width="1100" height="{h}" fill="#f2f8f5"/><g font-family="Noto Sans CJK SC, sans-serif">'+t(32,44,title,28)+body+t(32,h-24,foot,18)+'</g></svg>'
    (ROOT/'assets'/name).write_text(data+'\n')
write('loop-identity.svg','身份不变，运行集合和交付顺序不同',
 box(32,85,300,138,'输入 requests',['A → true','B → false','C → true'])+
 box(402,85,300,138,'完成顺序',['tick 4: A、C 完成','tick 5: B 完成','完成集合不是输出数组'])+
 box(772,85,295,138,'按 requests 取 output',['A → true','B → false','C → true'])+
 arrow(333,155,398,155)+arrow(703,155,768,155)+
 box(100,280,900,100,'running 可以删成员；requests 保留输入顺序',['输出保存在对应 Request 上，不靠完成事件的位置猜身份。']),
 '教学字符 oracle；并发完成顺序不决定调用者交付顺序。')
b=''
for i,(tick,parts,progress) in enumerate([(1,'A: 4','4 / 0 / 0'),(2,'A: 3 + B: 1','7 / 1 / 0'),(3,'B: 2 + C: 2','7 / 3 / 2'),(4,'C: 3（剩余1不用）','7 / 3 / 5')]):
 x=32+i*270;b+=box(x,100,250,150,f'tick {tick} · budget = 4',[parts,'computed A / B / C',progress])
 if i<3:b+=arrow(x+251,175,x+267,175)
b+=box(160,300,780,95,'computed == prompt_length 才移到 running',['take = min(prompt_length − computed, budget)，两个上界缺一不可。'])
write('prefill-budget.svg','预算沿队头消费：A(7)、B(3)、C(5)',b,'第02课先完成全部 prefill，再做5次 decode：共9个逻辑tick。')
b=box(32,90,300,130,'准入：需要 n 个块',['n > 可用总量 → NEVER','n > 当前空闲 → LATER','否则 → OK'])
b+=box(402,90,300,130,'块 i 的所有权',['allocate: refs[i] = 1','pin: 1 → 2','first free: 2 → 1'])
b+=box(772,90,295,130,'最后一个 owner 退出',['last free: 1 → 0','此时才可以重分配','再次 free 必须报错'])
b+=arrow(333,155,398,155)+arrow(703,155,768,155)
b+=box(100,280,900,108,'总块数8时只有7个可用：A占3、B占2、C需要3',['空闲2不足以放C → C等候；A完成释放3 → C可以继续。','block 0 保留；引用归零不是销毁底层 GPU 缓存张量。'])
write('block-ownership.svg','容量、身份、引用：三种信息不能压成一个计数器',b,'等待并不等于非法；正常结束和异常出口都要结清所有权。')
b=box(32,90,300,140,'第一票：是否尝试？',['ALL点：all(wants)','ANY点：any(wants)','重合时ANY优先'])
b+=box(402,90,300,140,'第二票：谁真的能做？',['actual = any(want and can)','必须按同一rank配对','真实上游用调度后的 outputs'])
b+=box(772,90,295,140,'共同的阶段',['vote 且 actual → prefill','否则 → decode','无本地工作也须保持步型'])
b+=arrow(333,160,398,160)+arrow(703,160,768,160)
b+=box(100,285,900,100,'反例：wants=[True, False]，can=[False, True]',['两个 any 都为True，却没有任何rank既想做又能做：actual=False。'])
write('parallel-votes.svg','意愿不是调度成功：两轮检查保护前进路径',b,'布尔向量是教学模型，不执行 collective，也不证明分布式无死锁。')
b=''
rows=[('total_q','expected','graph','cost','score'),('2','2.0','2','1.0','2.0000'),('3','2.9','4','1.3','2.2308'),('4 ★','3.8','4','1.3','2.9231'),('5','4.3','8','3.0','1.4333'),('6','4.8','8','3.0','1.6000')]
for r,row in enumerate(rows):
 y=90+r*43
 if r==3:b+='<rect x="45" y="198" width="990" height="41" rx="6" fill="#cceee3"/>'
 for c,v in enumerate(row):b+=t(65+c*200,y,v,22)
b+=t(55,380,'B、C各保留1个主位置；额外收益队列：0.9(B), 0.9(C), 0.5(B), 0.5(C)',19)
write('verify-budget.svg','容量台阶让“更多”不再等于“更好”',b,'选 total_q=4 → query_lengths=[2,2]；cost与score均为合成值，非真实性能。')
b=box(32,100,300,145,'计划：14个验证位置',['sum(query_lengths)','采样之前记账','B末尾计划2，实际剩1'])+box(402,100,300,145,'提交：13个字符',['len(true)+len(false)','+len(true) = 4+5+4','pending排空后才结清'])+box(772,100,295,145,'调度：15个逻辑tick',['包含prefill与decode','包含最终flush','不是15秒或15次kernel'])+arrow(333,175,398,175)
b+=box(100,300,900,90,'最终验收同时看内容与生命周期',['输出 true / false / true；pending=[]；活引用=0；finished含A/B/C。'])
write('metrics-ledger.svg','三本账：计划、交付、调度各用自己的单位',b,'14 / 15 不能命名 tokens/s：分子是位置，分母不是时间。')
print('Generated 6 core mechanism diagrams.')
