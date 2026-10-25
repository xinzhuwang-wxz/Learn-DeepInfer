"""Build the compute chapter's five SVG diagrams using only the standard library."""
from html import escape
from math import hypot
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INK = '#183b36'
MUTED = '#526963'
TEAL = '#087e70'
AMBER = '#98651b'
RED = '#a4473d'
BLUE = '#365c91'


def text(x, y, value, size=17, weight=400, color=INK):
    return f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}">{escape(value)}</text>'


def rect(x, y, width, height, fill='#fff', stroke='#bad2ca', radius=12):
    return f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" fill="{fill}" stroke="{stroke}"/>'


def panel(x, y, width, height, title, lines, color=TEAL):
    result = rect(x, y, width, height) + rect(x + 1, y + 1, 5, height - 2, color, color, 2)
    result += text(x + 18, y + 30, title, 19, 650)
    for row, line in enumerate(lines):
        result += text(x + 18, y + 61 + row * 27, line, 16)
    return result


def arrow(x1, y1, x2, y2, color=TEAL):
    length = hypot(x2 - x1, y2 - y1)
    ux, uy = (x2 - x1) / length, (y2 - y1) / length
    bx, by = x2 - 9 * ux, y2 - 9 * uy
    points = f'{x2},{y2} {bx - 4 * uy},{by + 4 * ux} {bx + 4 * uy},{by - 4 * ux}'
    return f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{color}" stroke-width="2.5"/><polygon points="{points}" fill="{color}"/>'


def save(name, title, description, width, height, body):
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">'
           f'<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>'
           '<g font-family="Noto Sans CJK SC, PingFang SC, Microsoft YaHei, sans-serif">'
           + rect(0, 0, width, height, '#f3f8f5', '#d8e5de', 18) + body + '</g></svg>')
    (ROOT / 'assets' / name).write_text(svg + '\n')


body = text(28, 42, 'query 降通道维；压缩 KV 缩短历史轴', 25, 700)
body += text(28, 74, 'N 是本次 query 行数；H、R、A、D、G、O 分别来自模型配置，不预设某个 checkpoint。', 16, color=MUTED)
body += panel(28, 108, 200, 138, 'hidden_states', ['[N, H]', '本层 attention 输入'])
body += arrow(233, 177, 282, 177) + text(240, 150, 'Wq.A', 15, 650)
body += panel(292, 108, 200, 138, 'q_a', ['[N, R]', 'R = q_lora_rank'])
body += arrow(497, 177, 580, 177) + text(505, 146, 'norm → B', 15, 650)
body += panel(590, 108, 208, 138, 'q_b', ['[N, A × D]', '可另带 index query'])
body += arrow(803, 177, 850, 177)
body += panel(860, 108, 212, 138, '主 attention Q', ['[N, A, D]', 'norm / RoPE 可融合'])
body += rect(28, 270, 1044, 58, '#fff5e5', '#d7b77c')
body += text(46, 305, 'Q 的 N 没因低秩投影减少；KV 的 ratio 才描述压缩历史的另一个维度。', 18, 600, AMBER)
body += panel(28, 352, 255, 138, 'attention context', ['语义形状 [N, A, D]', '分组 M = (A / G) × D'])
body += arrow(288, 420, 335, 420)
body += panel(345, 352, 292, 138, '分组 Wv', ['输入 [N, G, M]', '权重 [G, O, M]'])
body += arrow(642, 420, 689, 420)
body += panel(699, 352, 373, 138, '输出投影', ['[N, G, O] → [N, G × O]', 'out_proj → [N, H]'])
body += text(28, 530, '只核对形状的小例子：N=2, H=4, R=2, A=2, D=2, G=1, O=2。', 17, 600)
body += text(28, 562, '2×4 → 2×2 → 2×4 → 2×2×2；context 经 Wv 与 out_proj 回到 2×4。', 17)
body += text(28, 594, '这组小维度是纸上示例，不是 DeepInfer 可运行配置，也不是 CPU 实验中的模型权重。', 16, color=MUTED)
save('compute-query-path.svg', '低秩 query 与分组输出投影的形状', '查询从N乘H经过A投影变成N乘R，经过归一化和B投影变成N乘A乘D。注意力上下文按G组做Wv，然后输出投影回N乘H。低秩维R不等于KV压缩比。', 1100, 626, body)

body = text(28, 42, '形状仍是 4×2，V 却已经换了身份', 25, 700)
body += text(28, 74, 'CPU 参考：只把 V 分页；K 保持逻辑顺序。block_size = 2，block_table = [1, 0]。', 16, color=MUTED)
body += panel(28, 109, 290, 171, 'physical block 0', ['保存逻辑位置 2、3', 'V₂ = [2, 2]', 'V₃ = [99, 99]'], AMBER)
body += panel(28, 309, 290, 171, 'physical block 1', ['保存逻辑位置 0、1', 'V₀ = [1, 0]', 'V₁ = [0, 1]'])
body += panel(389, 109, 274, 171, '逻辑 block 0 → 1', ['读 physical block 1', '恢复逻辑位置 0、1', '这两行 visible = [0, 1]'])
body += panel(389, 309, 274, 171, '逻辑 block 1 → 0', ['读 physical block 0', '恢复逻辑位置 2、3', '本次不参加 attention'], AMBER)
body += arrow(324, 394, 381, 196) + arrow(324, 194, 381, 394, AMBER)
body += arrow(669, 196, 718, 196)
body += panel(728, 109, 344, 171, '正确聚合', ['p = [0.669762, 0.330238]', 'V = [[1, 0], [0, 1]]', 'out ≈ [0.669762, 0.330238]'])
body += panel(728, 309, 344, 171, '错用 block_table = [0, 1]', ['概率不变，但 V 读成 2、3', 'V = [[2, 2], [99, 99]]', 'out ≈ [34.033130, 34.033130]'], RED)
body += rect(28, 514, 1044, 76, '#e8f5ed')
body += text(46, 544, 'visible 决定“可看谁”；block_table 决定“这个谁存在哪里”。', 19, 650)
body += text(46, 573, '仅检查长度、二维 shape 和 softmax 归一性，拦不住这一类地址错误。', 16)
save('compute-paged-identity.svg', '正确与错误的分页地址映射', '物理块0保存逻辑位置2和3，物理块1保存逻辑位置0和1。正确块表1,0恢复V身份。错误块表0,1不改变维度但使attention结果变成34.033130。', 1100, 622, body)

body = text(28, 42, 'prefill 拼一个工作视图；decode 按索引读缓存', 25, 700)
body += text(28, 74, '两种接口都需要正确的可见集合，但索引对应的地址空间不同。', 17, color=MUTED)
body += text(28, 119, 'PREFILL · 示例行数', 19, 700, TEAL)
body += panel(28, 140, 230, 115, '历史 SWA', ['2 行 → kv[0:2]'])
body += panel(280, 140, 230, 115, '当前 SWA', ['2 行 → kv[2:4]'])
body += panel(532, 140, 230, 115, 'compressed cache', ['3 行 → kv[4:7]'])
body += arrow(770, 197, 812, 197)
body += panel(822, 140, 250, 115, '连续 BF16 工作视图', ['kv.shape = [7, 1, D]'])
body += text(28, 288, 'main_kv_offset = 2 + 2 = 4；compressed 第 1 行进入 kv 后是第 5 行。', 17, 600)
body += text(28, 318, 'kv_index_cat 把 SWA 与压缩索引放进同一工作视图；不能遗漏这个偏移。', 16, color=MUTED)
body += text(28, 369, 'DECODE · 保留缓存布局', 19, 700, BLUE)
body += panel(28, 392, 355, 135, 'SWA cache + kv_swa_index', ['按序列 state slot / 窗口定位', '不是 prefill 工作视图的行号'], BLUE)
body += panel(405, 392, 355, 135, 'ca_cache + kv_ca_index', ['按压缩页与物理索引定位', '连同 FlashMLA schedule metadata'], BLUE)
body += arrow(770, 456, 812, 456, BLUE)
body += panel(822, 392, 250, 135, 'decode kernel', ['读两套缓存 / 索引', '输出 context'], BLUE)
body += rect(28, 557, 1044, 77, '#fff5e5', '#d7b77c')
body += text(46, 588, 'YoCo compact：query 可变短，main / index KV 仍由 full hidden 更新。', 18, 650, AMBER)
body += text(46, 617, '图中 2 + 2 + 3 仅解释缓冲区地址；不是对某个请求的压缩输出数作预测。', 16)
save('compute-prefill-decode.svg', 'prefill与decode不同的缓存读取接口', 'Prefill把历史SWA、当前SWA和压缩缓存拼入连续BF16工作视图，压缩索引需要加主KV偏移。Decode使用SWA与压缩缓存各自的物理索引以及FlashMLA调度元数据。YoCo查询收缩不替代完整KV更新。', 1100, 666, body)

body = text(28, 42, 'data 和 scale 不只要一起数，还要按正确顺序找', 25, 700)
body += text(28, 74, '以下偏移都从 0 开始；区间含两端。示意框宽不按字节比例。', 16, color=MUTED)
body += text(28, 119, 'Main / SWA：每个 token 自带 data | scale', 19, 700, TEAL)
body += rect(28, 140, 666, 76, '#e8f5ed') + rect(694, 140, 378, 76, '#fff5e5', '#d7b77c')
body += text(47, 170, 'V41_FORMAT · 512 个 FP8 · 512B', 18, 650)
body += text(47, 199, 'data offset 0…511', 16)
body += text(712, 170, '16 个 UE8M0 · 16B', 18, 650, AMBER)
body += text(712, 199, 'scale offset 512…527；共 528B', 16)
body += rect(28, 239, 666, 76, '#e8f5ed') + rect(694, 239, 378, 76, '#fff5e5', '#d7b77c')
body += text(47, 269, 'V41_FP4_FORMAT · 512 个 FP4 · 256B', 18, 650)
body += text(47, 298, 'data offset 0…255', 16)
body += text(712, 269, '32 个 E4M3 · 32B', 18, 650, AMBER)
body += text(712, 298, 'scale offset 256…287；共 288B', 16)
body += text(28, 364, 'FP4 index：一个含 2 行的 block，先存全部 data，再存全部 scale', 19, 700, BLUE)
for x, w, title, sub, color in [(28,350,'token 0 data · 64B','offset 0…63',BLUE),(378,350,'token 1 data · 64B','offset 64…127',BLUE),(728,172,'token 0 scale','128…131 · 4B',AMBER),(900,172,'token 1 scale','132…135 · 4B',AMBER)]:
    body += rect(x, 390, w, 84, '#edf2fa' if color == BLUE else '#fff5e5')
    body += text(x+12, 423, title, 16, 650, color) + text(x+12, 454, sub, 15)
body += text(28, 511, '形状可以写成 [num_blocks, 2, 1, 68]；不能因此把每个 68B 切片当成完整 token。', 17, 600)
body += text(28, 545, 'split_kv_cache 按 block 划 data / scale；main / SWA 才按 token 拆两部分。', 17)
body += rect(28, 577, 1044, 70, '#fff5e5', '#d7b77c')
body += text(46, 606, '528、288、68 是相应格式的单行 payload，不是服务的“每 token 总显存”。', 18, 650, AMBER)
body += text(46, 633, '整块对齐、层数、压缩比、共享、FP32 压缩 state 和其它工作区还要另算。', 16)
save('compute-cache-bytes.svg', '缓存data与scale的两种排列', 'V41 FP8每行512字节数据加16字节scale，FP4 main每行256字节数据加32字节scale。两行FP4 index的块先放128字节数据再放8字节scale，4D uint8视图不意味着token交错记录。', 1100, 677, body)

body = text(28, 42, '选的是 logical expert，执行地址是 physical slot', 25, 700)
body += text(28, 74, '教学路由：3 个逻辑专家，1 个冗余副本，EP = 2，每个 rank 持有 2 个物理槽。', 16, color=MUTED)
body += panel(28, 110, 328, 198, '先选模型函数', ['scores = [0.6, 0.3, 0.1]', 'bias = [0, 0, 0.5]', '选 logical 0、2', '原分数归一化 → 6/7、1/7'])
body += arrow(362, 208, 408, 208)
body += panel(418, 110, 260, 198, '再查物理映射', ['logical 0 → [3]', 'logical 1 → [1]', 'logical 2 → [0, 2]', '本例各取第一个槽'])
body += arrow(684, 208, 730, 208)
body += panel(740, 110, 332, 198, 'rank / local', ['physical 3 → rank 1, local 1', 'physical 0 → rank 0, local 0', 'rank = physical // 2', 'local = physical % 2'])
body += panel(28, 342, 505, 149, 'rank 0', ['local 0 / physical 0：F₂ = [1, 1]  ← 选中', 'local 1 / physical 1：F₁ = [0, 2]'])
body += panel(555, 342, 517, 149, 'rank 1', ['local 0 / physical 2：F₂ = [1, 1]  ← 同函数副本', 'local 1 / physical 3：F₀ = [2, 0]  ← 选中'])
body += rect(28, 521, 1044, 88, '#e8f5ed')
body += text(46, 553, 'combine = x + (6/7) F₀ + (1/7) F₂', 21, 650)
body += text(46, 586, 'x = attention 输出；得到 [2.526904, 0.473096]。把 F₂ 从槽 0 移到槽 2，结果不变。', 17)
body += text(28, 645, '真实 gate 使用 sqrtsoftplus / noaux_tc；这里的给定分数与取首副本仅是教学规则。', 16, color=MUTED)
save('compute-expert-identity.svg', '逻辑专家物理副本与本地编号', '教学gate选逻辑专家0和2，归一化权重七分之六与七分之一，映射到物理槽3和0，分别由rank1 local1和rank0 local0执行。物理槽0和2都是逻辑函数2的副本，不应改变combine结果。', 1100, 677, body)

print('Wrote 5 compute chapter SVGs.')
