# 10 · 从教学实验，走向真实验证

## 一张能力分层表

| 层次 | 本教程提供与执行 | 能证明什么 | 不能证明什么 |
|---|---|---|---|
| 教学机制 | 9 个独立累积快照、单机制探针、自动测试 | 模拟器的离散状态不变量 | 模型输出质量与 kernel 正确性 |
| 源码证据 | 固定 commit、文件哈希、行号、调用链 | 讲解有可查的实现依据 | 上游在此环境可运行 |
| 上游 metadata / 单测 | 给出位置与条件，未执行 | 执行后才可报告对应断言 | 整体服务性能 |
| 真实离线推理 | 未构建、未下载权重、未执行 | 无运行结论 | 不能说“已复现 DeepInfer” |
| 生产服务 | 明确缺失外部组件 | 帮助识别架构边界 | 不提供完整服务部署 |

## 为什么这里不让你第一步运行 build.sh

上游 build.sh 不只是构建一个 Python 包：它安装平台相关依赖，NVIDIA 分支重装指定 NCCL，再构建 vendor 依赖与本体。examples 还可能自动下载默认 checkpoint，并在创建 LLM 时启用 trust_remote_code。理解这些副作用后才能在合适环境中执行。[源码 · 模型解析、环境设置与启动边界](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/examples/_config.py#L33-L125)

本项目没有运行任何上游构建、初始化 submodule、安装 GPU 包或下载权重。源码快照用于阅读；离线包仅保留 `reference/DeepInfer` 的实际引用文件与许可，不携带整个原始仓库。学习源码无需先承担大规模硬件与安装成本。

## 真机验证前的检查清单

1. 记录使用的 commit 与模型 revision，核对 checkpoint 的兼容架构与许可。
2. 核实真实硬件代际、卡数、拓扑、显存容量、驱动、CUDA/CANN、GCC、Python。
3. 阅读 build.sh、scripts/build_vendor.sh 与 requirements，预计网络、磁盘和编译资源。
4. 单独隔离环境，避免修改工作机器现有 NCCL / torch 环境。
5. 明确是否允许模型下载与执行远程模型代码。已有本地模型时，检查 config.json 与权重是否完整；不要只靠目录名。
6. 先做官方小型正确性路径，再扩展长上下文与结构化输出；通过后才做稳定测量。
7. 未达到 README 支持条件时，记录为未验证，而不是随意换卡后宣称支持。

卡数由具体实验决定。README 的性能图、Q&A 的高效生产单元、某个 examples 的 `--ep 8` 默认值不是同一个问题，不能压缩成一个“最小 8 卡 / 最小 128 卡”的统一结论。

## 继续读哪里

- 请求入口与生命周期：entrypoints/llm.py → engine/llm_engine.py → C++ scheduler。
- 模型编排：models/deepseek.py → decoder_layer.py → v4_modules，观察 V4 attention、MoE、YoCo、Engram 与 DFlash 的连接点。
- 执行与图捕获：worker/model_runner.py、cuda_graph_runner.py、dflash_cuda_graph_runner.py。
- 设备实现：model_executor/layers/tilelang_ops、npu_ops 与 vendor 子模块；算法说明不能替代硬件实现。
- 多模态：ve 的异步处理、image metadata 和失败请求进入 scheduler 的路径。
- 测试：test_batch_invariant、test_eplb_policy、test_offline_inference 及 long-context 路径。

这些是后续路线，不代表本书逐行覆盖了所有算子。V4 层直接依赖自定义 kernel 与模型模块，进一步证明纯 CPU 小例子只是教学抽取。[源码 · V4 层编排与自定义 kernel 依赖](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/models/decoder_layer.py#L1-L160)

## 一次可执行的源码核对

开始真机实验前，先确认阅读的文件没有漂移。此步骤不需要GPU：

```bash
python examples/audit_evidence.py
python examples/check_system.py
```

第一个报告当前索引的37个范围哈希与行界通过，并明确`gpu_correctness: not tested`；第二个报告9个教学阶段输出、末尾资源、失败清理和计数契约。两份结果不能合并成“DeepInfer已复现”。

审计器的核心不是尝试import上游，而是读取字节：

<!-- code: examples/audit_evidence.py:L12-L19 -->
```python
for key, item in sources.items():
    path = ROOT / 'reference' / 'DeepInfer' / item['path']
    content = path.read_bytes()
    assert hashlib.sha256(content).hexdigest() == item['sha256'], key
    lines = content.decode().splitlines()
    assert 1 <= item['start_line'] <= item['end_line'] <= len(lines), key
    assert '/blob/' + commit + '/' in item['url'], key
    checked.append(key)
```

每项依次检查文件SHA-256、行号范围、固定提交链接。行号有效只说明没越界，不代表这段恰好支持某条解释；仍要读上下文。哈希值也依赖可信基线，并不是数字签名或整个供应链安全认证。

### 从通过的检查，走到仍未执行的实验

```text
文件字节一致 ──→ 可定位的实现事实
                         │ 不能跳过硬件/软件/权重条件
                         ▼
设备最小正确性 ──→ 更长输入与结构化输出 ──→ 稳态性能
       │失败                  │失败                 │缺日志
       └── 保留复现输入和首个偏差，不填写“已通过” ────┘
```

每一级都需要自己的成功条件。最小正确性至少比较请求身份、输出长度与内容、错误状态；数值对照需说明容差或确定性条件；性能记录还需明确warmup、计时同步和统计窗口。不能用“进程退出0”替代所有这些验收。

例如一次推理输出正确，但日志缺少拓扑与稳态计时，应记录为“该输入的输出检查通过，性能未建立”；不是失败全部，也不是性能通过。反过来，漂亮的吞吐图也不能覆盖一次重复提交或资源泄漏。

### 练习与参考答案

1. 本地哈希都对，但没有驱动，哪一层通过？答：来源定位层；设备层未运行，不能填零吞吐当失败测量。
2. CPU输出13字符、verification_slots为14，应该与官方哪条tokens/s直接比较？答：都不能。先统一token单位、真实时间、接受口径和并行拓扑。
3. 打算使用本地权重，是否可忽略`trust_remote_code`？答：不可以；本地路径与是否执行模型代码是两项独立检查，需要审阅配置及代码来源。
4. 某次失败后的资源归零，是否算推理正确？答：只能说明该清理断言成立，必须继续检查异常种类和内容/身份契约。
5. 新commit仍满足行号范围，是否可以沿用旧结论？答：不可以；对比函数、调用者和默认条件，再重做对应实验。

## 术语速查

- prefill：处理输入上下文的新计算；decode：逐步生成输出。
- query 长度：当前 step 处理的位置数；key 长度：attention 可用历史范围。
- KV block：缓存的分配单位；state：某些固定窗口或模型相关状态，不能随意等同普通 block。
- rank：分布式参与者；DP/CP/EP 分别针对数据、上下文、专家组织工作。
- in-flight：已开始但尚未完成落账的工作；drain/flush：清空管线尾部。
- batch invariant：不同 batch 组成下保持相同结果的条件；不等于仅固定随机 seed。
- full-acceptance accounting：按所有候选接受的口径统计，不等于真实交付计数。
