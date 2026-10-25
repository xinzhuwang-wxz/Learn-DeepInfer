# 13 · 版本、许可与来源

本教程是独立教育材料，不是 DeepSeek 官方文档或产品。源码核验于 2026-10-25。Markdown 是正文来源；网页和完整代码册由课程清单与配套代码生成。

## 固定源码

DeepInfer commit：`9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`。[上游 Apache-2.0 许可证原文](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/LICENSE)的副本随项目及网页版附带；原创讲解、图、代码与发布脚本使用 MIT。upstream 源码的原有声明保持不变，依赖子模块未初始化。

## 逐项证据索引

| 主题 | 文件与行号 | 说明 |
|---|---|---|
| [decoder 层的 norm / attention / MoE 编排](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/models/decoder_layer.py#L112-L182) | deepinfer/model_executor/models/decoder_layer.py:112-182 | 固定提交中的实现或文档 |
| [低秩 query、压缩 KV 与精度容器](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/models/v4_modules/mla_attn.py#L26-L229) | deepinfer/model_executor/models/v4_modules/mla_attn.py:26-229 | 固定提交中的实现或文档 |
| [prefill / decode attention 与缓存数据流](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/models/v4_modules/attn.py#L589-L880) | deepinfer/model_executor/models/v4_modules/attn.py:589-880 | 固定提交中的实现或文档 |
| [KVCache 与 StateBlock 物理布局](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/worker/cache_engine.py#L206-L242) | deepinfer/worker/cache_engine.py:206-242 | 固定提交中的实现或文档 |
| [FP8 / FP4 data 与 scale 字节格式](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/tilelang_ops/cache_common.py#L24-L162) | deepinfer/model_executor/layers/tilelang_ops/cache_common.py:24-162 | 固定提交中的实现或文档 |
| [logical / physical expert 与门控参数](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/models/v4_modules/moe.py#L1-L112) | deepinfer/model_executor/models/v4_modules/moe.py:1-112 | 固定提交中的实现或文档 |
| [精度选择、gate、dispatch、grouped GEMM、combine](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/models/v4_modules/mlp.py#L149-L616) | deepinfer/model_executor/models/v4_modules/mlp.py:149-616 | 固定提交中的实现或文档 |
| [FP8 cast、scale layout 与输出 stride](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/linear.py#L10-L174) | deepinfer/model_executor/layers/linear.py:10-174 | 固定提交中的实现或文档 |
| [发布范围、硬件与复现入口](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/README.md#L1-L96) | README.md:1-96 | 固定提交中的实现或文档 |
| [LLM.generate → _run_engine](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/entrypoints/llm.py#L93-L198) | deepinfer/entrypoints/llm.py:93-198 | 固定提交中的实现或文档 |
| [模型解析、环境设置与启动边界](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/examples/_config.py#L33-L125) | examples/_config.py:33-125 | 固定提交中的实现或文档 |
| [LLMEngine.step / flush / post_process](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/engine/llm_engine.py#L860-L1029) | deepinfer/engine/llm_engine.py:860-1029 | 固定提交中的实现或文档 |
| [PD 与外部缓存扩展钩子](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/engine/llm_engine.py#L800-L829) | deepinfer/engine/llm_engine.py:800-829 | 固定提交中的实现或文档 |
| [NORMAL 引擎的两轮投票](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/scheduler.hpp#L275-L346) | csrc/engine/scheduler.hpp:275-346 | 固定提交中的实现或文档 |
| [_schedule_prefill](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/scheduler.hpp#L351-L554) | csrc/engine/scheduler.hpp:351-554 | 固定提交中的实现或文档 |
| [_schedule_decode / schedule](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/scheduler.hpp#L556-L735) | csrc/engine/scheduler.hpp:556-735 | 固定提交中的实现或文档 |
| [Blocks / BlockManager 引用计数](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/block_manager.hpp#L1-L178) | csrc/engine/block_manager.hpp:1-178 | 固定提交中的实现或文档 |
| [Forward → Post-process → Sample](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/worker/worker.py#L89-L189) | deepinfer/worker/worker.py:89-189 | 固定提交中的实现或文档 |
| [prepare_zos_inputs 请求重映射](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/zos.cpp#L142-L187) | csrc/engine/zos.cpp:142-187 | 固定提交中的实现或文档 |
| [MTPVarlen 的预算目标与置信度时序](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/mtp_varlen.hpp#L135-L275) | csrc/engine/mtp_varlen.hpp:135-275 | 固定提交中的实现或文档 |
| [副本分配与逐步峰值贪心布局](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/eplb/policy.py#L1-L140) | deepinfer/model_executor/eplb/policy.py:1-140 | 固定提交中的实现或文档 |
| [SWA 可见域与 keep_tokens 约束](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/input_metadata/prepare_tensor_input_v4.cpp#L129-L359) | csrc/engine/input_metadata/prepare_tensor_input_v4.cpp:129-359 | 固定提交中的实现或文档 |
| [mask / compact metadata 视图](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/input_metadata/yoco.py#L68-L182) | deepinfer/model_executor/input_metadata/yoco.py:68-182 | 固定提交中的实现或文档 |
| [L1 metadata / L2 数值等价验证](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/examples/yoco/README.md#L1-L41) | examples/yoco/README.md:1-41 | 固定提交中的实现或文档 |
| [mask、accept_token、rollback](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/structure_output/matcher.py#L118-L245) | deepinfer/structure_output/matcher.py:118-245 | 固定提交中的实现或文档 |
| [异步语法编译、入队与重放](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/structure_output/so_matcher_builder.py#L58-L129) | deepinfer/structure_output/so_matcher_builder.py:58-129 | 固定提交中的实现或文档 |
| [SamplingParams 与 seq_seed](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/sampling_params.py#L13-L119) | deepinfer/sampling_params.py:13-119 | 固定提交中的实现或文档 |
| [单条与批量输出逐项比较](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/tests/models/test_batch_invariant.py#L1-L116) | tests/models/test_batch_invariant.py:1-116 | 固定提交中的实现或文档 |
| [稳态指标与满接受率口径](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/examples/benchmark.py#L380-L412) | examples/benchmark.py:380-412 | 固定提交中的实现或文档 |
| [配置一致性与 VARLEN 重标定限制](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/examples/sweep/plot_pareto.py#L22-L78) | examples/sweep/plot_pareto.py:22-78 | 固定提交中的实现或文档 |
| [生产系统背景，不等于发布内容](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/doc/inference_deployment_qa.md#L1-L130) | doc/inference_deployment_qa.md:1-130 | 固定提交中的实现或文档 |
| [V4 层编排与自定义 kernel 依赖](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/models/decoder_layer.py#L1-L160) | deepinfer/model_executor/models/decoder_layer.py:1-160 | 固定提交中的实现或文档 |

| [模型与采样的发起边界](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/worker/model_runner.py#L308-L367) | deepinfer/worker/model_runner.py:308-367 | 固定提交中的实现或文档 |
| [event门控的后处理与尾部排空](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/worker/worker.py#L248-L283) | deepinfer/worker/worker.py:248-283 | 固定提交中的实现或文档 |
| [RequestOutput.clone 复制记录](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/pybind_output.cpp#L16-L18) | csrc/engine/pybind_output.cpp:16-18 | 固定提交中的实现或文档 |
| [输出池记录的分配与释放](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/csrc/engine/outputs.hpp#L45-L80) | csrc/engine/outputs.hpp:45-80 | 固定提交中的实现或文档 |
| [分资源的延迟释放与clone unpin](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/engine/llm_engine.py#L92-L116) | deepinfer/engine/llm_engine.py:92-116 | 固定提交中的实现或文档 |

## 验证结论的边界

教学测试运行在 CPU，仅覆盖原创模拟器。上游源码从未执行；没有构建 GPU 扩展、下载模型、运行真实离线推理或复现性能图。README / Q&A 的规模数据仅用于说明其官方实验或部署条件，不是本教程测得的结果。

## 证据可更新，版本不能悄悄漂移

所有源码链接使用固定提交而非 main。sources.json 保存逐文件 SHA-256；修改上游文件后校验会失败。发布新版本时须重新阅读变化、更新教程与测试，然后显式更新锁文件。
