#!/usr/bin/env python3
"""Record already-inspected upstream files, without importing/executing them."""
from pathlib import Path
import hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]; UP=ROOT/'upstream/DeepInfer'
COMMIT='9bd9cf243fc53ef9afbf8f04d391d989117d6ab9'
assert subprocess.check_output(['git','-C',str(UP),'rev-parse','HEAD'],text=True).strip()==COMMIT
specs=[
('model_flow','deepinfer/model_executor/models/decoder_layer.py',112,182,'decoder 层的 norm / attention / MoE 编排'),
('attn_container','deepinfer/model_executor/models/v4_modules/mla_attn.py',26,229,'低秩 query、压缩 KV 与精度容器'),
('attn_runtime','deepinfer/model_executor/models/v4_modules/attn.py',589,880,'prefill / decode attention 与缓存数据流'),
('cache_layout','deepinfer/worker/cache_engine.py',206,242,'KVCache 与 StateBlock 物理布局'),
('cache_format','deepinfer/model_executor/layers/tilelang_ops/cache_common.py',24,162,'FP8 / FP4 data 与 scale 字节格式'),
('moe_container','deepinfer/model_executor/models/v4_modules/moe.py',1,118,'logical / physical expert 与门控参数'),
('moe_runtime','deepinfer/model_executor/models/v4_modules/mlp.py',149,621,'精度选择、gate、dispatch、grouped GEMM、combine'),
('linear_quant','deepinfer/model_executor/layers/linear.py',10,174,'FP8 cast、scale layout 与输出 stride'),
('scope','README.md',1,96,'发布范围、硬件与复现入口'),
('entry','deepinfer/entrypoints/llm.py',93,214,'LLM.generate → _run_engine'),
('launch','examples/_config.py',33,145,'模型解析、环境设置与启动边界'),
('engine','deepinfer/engine/llm_engine.py',860,1029,'LLMEngine.step / flush / post_process'),
('hooks','deepinfer/engine/llm_engine.py',800,829,'PD 与外部缓存扩展钩子'),
('votes','csrc/engine/scheduler.hpp',275,346,'NORMAL 引擎的两轮投票'),
('prefill','csrc/engine/scheduler.hpp',351,554,'_schedule_prefill'),
('decode','csrc/engine/scheduler.hpp',556,735,'_schedule_decode / schedule'),
('blocks','csrc/engine/block_manager.hpp',1,178,'Blocks / BlockManager 引用计数'),
('worker','deepinfer/worker/worker.py',89,189,'Forward → Post-process → Sample'),
('zos','csrc/engine/zos.cpp',142,187,'prepare_zos_inputs 请求重映射'),
('mtp','csrc/engine/mtp_varlen.hpp',135,281,'MTPVarlen 的预算目标与置信度时序'),
('eplb','deepinfer/model_executor/eplb/policy.py',1,145,'副本分配与逐步峰值贪心布局'),
('yoco','csrc/engine/input_metadata/prepare_tensor_input_v4.cpp',129,359,'SWA 可见域与 keep_tokens 约束'),
('yoco_view','deepinfer/model_executor/input_metadata/yoco.py',68,182,'mask / compact metadata 视图'),
('yoco_test','examples/yoco/README.md',1,50,'L1 metadata / L2 数值等价验证'),
('grammar','deepinfer/structure_output/matcher.py',118,245,'mask、accept_token、rollback'),
('builder','deepinfer/structure_output/so_matcher_builder.py',58,129,'异步语法编译、入队与重放'),
('sampling','deepinfer/sampling_params.py',13,119,'SamplingParams 与 seq_seed'),
('invariant','tests/models/test_batch_invariant.py',1,119,'单条与批量输出逐项比较'),
('benchmark','examples/benchmark.py',380,412,'稳态指标与满接受率口径'),
('pareto','examples/sweep/plot_pareto.py',22,78,'配置一致性与 VARLEN 重标定限制'),
('qa','doc/inference_deployment_qa.md',1,130,'生产系统背景，不等于发布内容'),
('model','deepinfer/model_executor/models/decoder_layer.py',1,160,'V4 层编排与自定义 kernel 依赖')]
result={}
for id,path,start,end,label in specs:
    raw=(UP/path).read_bytes();lines=raw.decode().splitlines();end=min(end,len(lines))
    result[id]={'label':label,'repository':'deepseek-ai/DeepInfer','commit':COMMIT,'path':path,'start_line':start,'end_line':end,'sha256':hashlib.sha256(raw).hexdigest(),'url':f'https://github.com/deepseek-ai/DeepInfer/blob/{COMMIT}/{path}#L{start}-L{end}'}
(ROOT/'sources.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
(ROOT/'upstream.lock.json').write_text(json.dumps({'repository':'https://github.com/deepseek-ai/DeepInfer','commit':COMMIT,'observed_utc':'2026-10-25','license':'Apache-2.0','source_files':len(result),'submodules_initialized':False,'upstream_code_executed':False},indent=2)+'\n')
print('Locked',len(result),'source anchors.')
