from pathlib import Path
from html import escape
ROOT=Path(__file__).resolve().parents[1]
def svg(title,width,height,body):
 return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}"><title>{escape(title)}</title><defs><marker id="arr" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L6,3z" fill="#087a64"/></marker></defs><rect width="100%" height="100%" rx="16" fill="#f0f7f3"/>'+body+'</svg>'
def text(x,y,s,size=17,color='#17342e',weight='400'):
 return f'<text x="{x}" y="{y}" font-family="Noto Sans CJK SC, sans-serif" font-size="{size}" fill="{color}" font-weight="{weight}">{escape(s)}</text>'
def box(x,y,w,h,title,sub=''):
 return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="white" stroke="#b9d3c7"/>'+text(x+16,y+30,title,18,weight='600')+(text(x+16,y+56,sub,12,'#526b64') if sub else '')
def line(x1,y1,x2,y2):return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#087a64" stroke-width="2" marker-end="url(#arr)"/>'
body=text(28,38,'同一批请求，每次只增加一个机制',21,weight='600')
names=[('01 请求身份','输出顺序不变量'),('02 分块预算','等待 → 运行'),('03 块所有权','暂时满 / 永远放不下'),('04 阶段投票','意愿 ≠ 可执行'),('05 验证预算','收益 / 成本'),('06 延迟提交','pending → flush'),('07 有界重放','保留可见域'),('08 语法回滚','token 与状态一起退'),('09 测量口径','验证量 ≠ 交付量')]
for i,(a,b) in enumerate(names):
 x=28+(i%3)*285;y=65+(i//3)*110;body+=box(x,y,260,78,a,b)
 if i%3<2:body+=line(x+263,y+39,x+280,y+39)
body+=text(28,407,'教学运行：Python 标准库 · 确定性 oracle · 无设备 · 不报告真实性能',14,'#526b64')
(ROOT/'assets/learning-path.svg').write_text(svg('九个递进机制',900,435,body))
body=text(28,36,'从 Python API 到设备执行的主路径',21,weight='600')
for x,y,w,t,sub in [(28,65,258,'LLM.generate','请求 ID / 参数复制 / 输出归序'),(329,65,258,'LLMEngine.step','调度 / 生命周期 / 后处理'),(630,65,240,'C++ Scheduler','prefill / decode / block'),(28,192,258,'Model + sampler','V4 / DFlash / kernel'),(329,192,258,'Worker / ModelRunner','forward / previous post / sample'),(630,192,240,'Metadata','query / key / cache layout')]:body+=box(x,y,w,80,t,sub)
for a in [(290,105,325,105),(590,105,625,105),(750,148,750,185),(625,232,594,232),(325,232,290,232),(155,277,155,306)]:body+=line(*a)
body+=text(28,337,'公开引擎内：请求、调度、设备缓存、模型与采样',16)+text(28,372,'公开版外部缺口：完整 KV cache 服务 / E-P-D 编排 / DP 负载均衡',16,'#975e31')
(ROOT/'assets/request-flow.svg').write_text(svg('请求路径与公开边界',900,400,body))
body=text(28,37,'ZOS 的逻辑次序：不是并发性能测量',21,weight='600')
for x,y,w,t,sub in [(30,73,260,'当前 forward N','设备工作可与 CPU 操作重叠'),(330,73,260,'上一轮 post N-1','提交结果、更新状态'),(630,73,240,'当前 sample N','保留 prev_step_state'),(30,202,260,'下一轮 forward N+1','通过 seq_id 对应输入'),(330,202,260,'post N','此时才交付上一轮结果'),(630,202,240,'尾部 flush','无新工作也必须 drain')]:body+=box(x,y,w,84,t,sub)
for a in [(295,115,324,115),(595,115,624,115),(295,244,324,244),(595,244,624,244)]:body+=line(*a)
body+=text(30,335,'四个不同状态：已调度 → 已采样 → 已提交 → 已交付',18)+text(30,370,'对象池复用前复制输出；释放资源必须服从完成事件与所有权。',16,'#526b64')
(ROOT/'assets/async-timeline.svg').write_text(svg('延迟提交和flush',900,400,body))
body=text(28,37,'同一层的三张账：shape / address / precision',21,weight='600')
for x,y,w,t,sub in [(28,70,260,'Hidden states','N × H · norm / HC'),(329,70,258,'Query 低秩路径','N × R → N × (heads · D)'),(630,70,240,'可见域与 cache','SWA + compressed / index'),(28,200,260,'回到 hidden space','context → Wv / O projection'),(329,200,258,'MoE：逻辑 → 物理','gate → dispatch → experts'),(630,200,240,'Combine + shared','归还原 token 位置')]:body+=box(x,y,w,80,t,sub)
for a in [(293,110,322,110),(592,110,623,110),(293,240,322,240),(592,240,623,240)]:body+=line(*a)
body+='<path d="M750 154 V180 H158 V196" fill="none" stroke="#087a64" stroke-width="2" marker-end="url(#arr)"/>'
body+=text(28,335,'数据与 scale 一起传；物理副本不能改变 logical expert 的函数。',17)+text(28,370,'参考实验只算小矩阵与字节账，不包含 V4 压缩、低精度 kernel 或通信。',14,'#526b64')
(ROOT/'assets/compute-path.svg').write_text(svg('计算路径形状地址精度账',900,400,body))
