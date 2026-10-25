"""Draw original replay/grammar diagrams with the standard library."""
from html import escape
from math import hypot
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, TEAL, AMBER, RED = '#183b36', '#526963', '#087e70', '#98651b', '#a4473d'

def text(x, y, value, size=17, weight=400, fill=INK):
    return f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}">{escape(value)}</text>'

def rect(x, y, w, h, fill='#fff', stroke='#bad2ca', rx=10):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>'

def arrow(x1, y1, x2, y2, color=TEAL):
    length = hypot(x2-x1, y2-y1)
    ux, uy = (x2-x1)/length, (y2-y1)/length
    bx, by = x2-9*ux, y2-9*uy
    points = f'{x2},{y2} {bx-4*uy},{by+4*ux} {bx+4*uy},{by-4*ux}'
    return f'<path d="M{x1},{y1} L{x2},{y2}" stroke="{color}" stroke-width="2.5" fill="none"/><polygon points="{points}" fill="{color}"/>'

def panel(x, y, w, h, title, lines, accent=TEAL):
    result = rect(x,y,w,h)+rect(x+1,y+1,5,h-2,accent,accent,2)+text(x+19,y+32,title,20,650)
    for i, line in enumerate(lines):
        result += text(x+19,y+64+i*28,line,16)
    return result

def save(name, title, desc, width, height, body):
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc"><title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc><g font-family="Noto Sans CJK SC, PingFang SC, Microsoft YaHei, sans-serif">{rect(0,0,width,height,"#f3f8f5","#d8e5de",18)}{body}</g></svg>'
    (ROOT/'assets'/name).write_text(svg+'\n')

# The three lower bounds, not a physical KV eviction picture.
body = text(28,42,'保留后缀，不等于每一行都能看见整个后缀',25,700)
body += text(28,74,'data_len = 12 · window = 4 · keep = 6 · backtrack = 2 · 位置从 0 开始',16,400,MUTED)
body += text(28,126,'绝对位置',17,650)
for p in range(12):
    x=248+p*56
    body += text(x+17,126,str(p),17,650)
rows = [
    (146,'保留矩形',set(range(6,12)),AMBER,'#fff1d7'),
    (210,'pos = 11 的滑窗',set(range(8,12)),TEAL,'#d8f0e8'),
    (274,'回退 2 后：pos = 9',set(range(6,10)),TEAL,'#d8f0e8'),
    (338,'pos = 11，valid = 10',set(range(10,12)),RED,'#f9e3dc'),
]
for y,label,active,accent,fill in rows:
    body += text(28,y+32,label,17,600)
    for p in range(12):
        x=248+p*56
        body += rect(x,y,46,45,fill if p in active else '#eef2ef',accent if p in active else '#d5dfd8',5)
        if p in active: body += text(x+17,y+29,str(p),16,650,accent)
body += panel(28,415,444,151,'为什么 keep 不能只有 5？',['keep=5 只保留 7…11','回退到 pos=9 仍需窗口 6…9','位置 6 已缺失，不能靠 mask 补回来。'],AMBER)
body += panel(500,415,444,151,'valid_start 是另一条边界',['valid_start=10 排除所有位置 <10','即使 8、9 在滑窗与保留后缀里','也不能读取已经失效的 state。'],RED)
body += text(28,603,'可见范围 = [ max(0, pos−window+1, data_len−keep, valid_start), pos+1 )',17,650)
body += text(28,635,'左界超过右界时裁成空集。这里画的是整数位置，不是显存分配、KV 内容或 attention 数值。',16,400,MUTED)
save('replay-bounds.svg','滑窗、保留后缀和有效状态的三个下界','长度12、窗口4、保留6。当前位置11看8至11，回退两步后位置9仍需6至9；保留5会缺位置6。有效状态从10开始时，位置11只可读10和11。',976,666,body)

body = text(28,42,'同一可见关系，两个行布局',25,700)
body += text(28,74,'教学 compare()：整段 12 个位置、保留 6 个位置；没有 tensor 或实际 attention。',16,400,MUTED)
body += panel(28,104,436,147,'full：保留行号 0…11',['0…5 的可见集合为空','6…11 保存对应的 key 位置集合','此处 full 是“全行 + 同一截断公式”'])
body += panel(512,104,436,147,'compact：只列行号 6…11',['行号仍是原绝对位置','不是把位置 6 改名成位置 0','检查保留行的可见集合相同'])
body += arrow(468,179,505,179)
for i,(p,keys) in enumerate([(6,'6'),(7,'6, 7'),(8,'6, 7, 8'),(9,'6, 7, 8, 9'),(10,'7, 8, 9, 10'),(11,'8, 9, 10, 11')]):
    y=274+i*42
    body += rect(28,y,920,35,'#fff','#d7e4dd',6)
    body += text(48,y+24,f'full[{p}]',16,650)+text(240,y+24,keys,16)
    body += text(492,y+24,'=',18,650,TEAL)+text(568,y+24,f'compact[{p}]',16,650)
body += rect(28,548,920,64,'#fff5e5','#d7b77c')
body += text(47,575,'这只是同一函数的投影自检，不是独立实现之间的数值等价证明。',18,650,AMBER)
body += text(47,599,'没有证明 store slot、压缩 key、CP 重排或模型输出相同。',16,400,MUTED)
body += text(28,651,'上游 compact 要一起更换：positions · cu_seqlens · SWA pivots · store mapping · 压缩 key 范围',16,650)
body += text(28,684,'真实分 chunk 还有占位行；CP>1 还要重分布。不能只对 hidden_states 切片。',16,400,MUTED)
save('replay-layout.svg','full行与compact行的可见集合对应','教学full包含十二行，compact包含后六行，同一绝对位置的可见集合相等。这里只做同一函数的投影自检。上游还要同步更新位置、变长序列边界、SWA pivots、存储映射、压缩key范围以及CP布局。',976,712,body)

body = text(28,42,'matcher 保存的是前缀，不是最后一次的允许集合',25,700)
body += text(28,74,'字符级语言只有 true / false。每次 allowed() 都由当前 prefix 重新计算。',16,400,MUTED)
# Nodes include the prefix and next-character set.
def node(x,y,prefix,allowed,complete=False):
    return rect(x,y,128,82,'#e1f2e9' if complete else '#fff',TEAL if complete else '#bad2ca')+text(x+15,y+30,prefix,20,650)+text(x+15,y+60,allowed,15,400,MUTED)
body += node(28,218,'空串','允许 t / f')
body += node(220,126,'t','允许 r')+node(408,126,'tr','允许 u')+node(596,126,'tru','允许 e')+node(784,126,'true','允许 <eos>',True)
body += node(220,310,'f','允许 a')+node(386,310,'fa','允许 l')+node(552,310,'fal','允许 s')+node(718,310,'fals','允许 e')+node(884,310,'false','允许 <eos>',True)
body += arrow(159,239,211,172)+arrow(159,276,211,348)
for x in (352,540,728): body += arrow(x,168,x+48,168)
for x in (352,518,684,850): body += arrow(x,351,x+28,351)
body += text(28,436,'撤销草稿 ru：prefix 从 tru 回到 t，允许集合从 {e} 恢复为 {r}。',21,650)
body += panel(28,464,464,145,'正确：输出与 matcher 一起撤回',["输出：tru → t", "matcher：tru → t",'下一轮 r 可以被接受。'])
body += panel(520,464,492,145,'错误：只撤销输出',["输出：tru → t", "matcher：仍停在 tru",'下一轮 r 被错误拒绝，因为它还在等 e。'],RED)
body += text(28,650,'单字符拒绝不改变 prefix；rollback(0) 不改变 prefix；越界回退必须显式报错。',17,600)
body += text(28,684,'<eos> 在这个 toy 中不写入历史，不记录终止状态；不能用它验证真实 EOS 后的回退。',16,400,MUTED)
save('grammar-state.svg','布尔字符语法状态机与回退','初始允许t或f，两条路径分别形成true和false。true路径的tru只允许e；回退两个字符后变为t并允许r。只撤销输出而不回退matcher会错误拒绝r。EOS不记录在教学历史中。',1040,712,body)

body = text(28,42,'同一批 A/B/C：pending 还没有推进语法状态',25,700)
body += text(28,74,'第13步 decode 返回后，C 的 tr 已经在途，但它的 output 与 matcher.prefix 都还是空串。',16,400,MUTED)
body += panel(28,110,282,170,'第13步结束 · C',["pending: 'tr'","output: ''","matcher.prefix: ''"],AMBER)
body += arrow(314,195,351,195)
body += panel(360,110,282,170,'第14步提交 tr',["accept('t') → 't'","accept('r') → 'tr'","output 追加 'tr'"])
body += arrow(646,195,683,195)
body += panel(692,110,282,170,'第15步 flush ue',["accept('u'), accept('e')","output: 'true'","matcher.prefix: 'true'"])
body += rect(28,313,946,58,'#e9f3ef','#c8dbd2')
body += text(47,347,'教学主线：用正确 oracle 产出字符，在 emit 前检查；没有草稿拒绝，也不调用 rollback。',17,600)
body += text(28,416,'上游 sampler：为了制作草稿各位置的 mask，临时走进未来，再恢复',22,650)
body += panel(28,449,282,150,'持久状态：已提交前缀',['接收本轮已确认的 token','先制作下一位置的 mask','记住这次进入前的状态'])
body += arrow(314,524,351,524)
body += panel(360,449,282,150,'临时推进草稿',['按顺序 accept draft','每步制作对应 vocab mask','拒绝后停止继续推进'],AMBER)
body += arrow(646,524,683,524)
body += panel(692,449,282,150,'恢复基线',['rollback(实际接受步数)','不是回退全部草稿长度','下一轮由真正接受结果推进'])
body += text(28,640,'两条路径不能混称：教学 emit 检查 ≠ 上游采样前 mask + 推测状态回退。',17,650,RED)
save('grammar-commit.svg','教学提交与上游草稿mask准备的不同边界','教学C请求第13步pending含tr但output和matcher仍为空；第14步提交tr推进语法，第15步flush ue完成。上游sampler则先接收实际提交token，临时接收草稿生成mask，再按实际临时接受步数回退。',1002,670,body)
print('Wrote 4 replay/grammar SVG diagrams.')
