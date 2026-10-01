# 大模型推理性能指标术语表 (LLM Inference Performance Terminology)

在 AI Infra 与大模型服务评估中，单靠“总生成时间”无法客观评估系统性能。本文档标准化定义了大模型推理性能评估的**核心三大维度**：单请求延迟指标、系统吞吐量指标与统计分位数。

---

## 1. 单请求延迟指标 (Per-Request Latency Metrics)

### 1.1 TTFT (Time To First Token / 首 Token 延迟)

* **定义**：从客户端/用户发起请求开始，到接收到系统返回的**第一个输出 Token** 所经过的完整时长。
* **链路构成**：

$$\text{TTFT} = T_{\text{queue}} + T_{\text{tokenize}} + T_{\text{prefill}} + T_{\text{schedule}} + T_{\text{sample\_1}}$$


* **网络与排队 ($T_{\text{queue}}$)**：网络传输及在服务端等待 Batch 调度的排队时间；
* **分词 ($T_{\text{tokenize}}$)**：文本转化为 Input Token IDs 的时间；
* **预填阶段 ($T_{\text{prefill}}$)**：模型对完整 Prompt 进行前向传播并生成 KV Cache 的时间；
* **采样与返回 ($T_{\text{sample\_1}}$)**：首个 Token 的采样及网络流式（Streaming）吐回时间。


* **测试范围说明**：
> 💡 **工程注意**：在模型级（Model-Only / Benchmark）测试中，通常只测量 $\text{Prefill} + \text{1st Token Sampling}$ 时间；而在系统级服务（Serving Engine）测试中，必须包含网络排队与 Tokenization。报告中需明确说明测试边界。


* **计算示例**：
* **场景**：用户发送一段 512 Tokens 的 Prompt，服务端完成 Tokenization、Prefill 计算并在 **120 ms** 后吐出第一个字。
* **指标**：$\text{TTFT} = 120 \text{ ms}$。



---

### 1.2 TPOT (Time Per Output Token / 输出 Token 平均延迟)

* **定义**：首 Token 吐出之后，在 Decode 阶段**生成后续每个 Output Token 所消耗的平均时间**。
* **计算公式**：

$$\text{TPOT} = \frac{\text{Total Time} - \text{TTFT}}{\text{Output Tokens} - 1}$$


* **计算示例**：
* **场景**：一次请求的总耗时为 $1120 \text{ ms}$，其中 $\text{TTFT} = 120 \text{ ms}$，最终共输出了 $51$ 个 Output Tokens。
* **计算**：

$$\text{TPOT} = \frac{1120 \text{ ms} - 120 \text{ ms}}{51 - 1} = \frac{1000 \text{ ms}}{50} = 20 \text{ ms/token}$$


* **指标**：$\text{TPOT} = 20 \text{ ms/token}$（即 Decode 阶段平均每秒生成 50 个 Token）。



---

### 1.3 ITL (Inter-Token Latency / Token 间延迟)

* **定义**：流式输出（Streaming）过程中，**相邻两个 Token 返回之间的精确时间间隔**。
* **与 TPOT 的区别**：
* **TPOT** 是一个**全局平均值**；
* **ITL** 是一个**时间序列/分布**，能够精确捕获生成过程中的局部卡顿与抖动。


* **计算示例**：
* **场景**：某一请求生成 5 个新 Token，各 Token 返回的时间戳间隔数组为 $[20\text{ms}, 19\text{ms}, 48\text{ms}, 18\text{ms}]$。
* **分析**：虽然平均 $\text{TPOT} = 26.25 \text{ ms}$，但第 3 个 Token 的 $\text{ITL} = 48 \text{ ms}$，反映出系统在该时刻可能发生了长 Token 调度、垃圾回收（GC）或显存重分配引起的局部卡顿。



---

## 2. 系统吞吐量指标 (System Throughput Metrics)

衡量推理系统总体处理能力时，必须严格区分以下三类吞吐量：

| 指标名称 | 英文名称 | 单位 | 衡量阶段与关注点 |
| --- | --- | --- | --- |
| **请求吞吐量** | Request Throughput | `requests/s` (RPS) | 系统整体高并发服务能力。 |
| **输入 Token 吞吐量** | Input Token Throughput | `input tokens/s` | Prefill 阶段的算力消耗（Compute-Bound 性能）。 |
| **输出 Token 吞吐量** | Output Token Throughput | `output tokens/s` | Decode 阶段的显存带宽与并发生成能力（Memory-Bound 性能）。 |

### 2.1 示例与计算

假设一个推理 Server 在 **10 秒** 的测试窗口内，并发处理并完成了 **50 个请求**，每个请求平均输入 Prompt 为 $512 \text{ tokens}$，平均输出为 $100 \text{ tokens}$：

1. **Request Throughput**：

$$\text{RPS} = \frac{50 \text{ requests}}{10 \text{ s}} = 5.0 \text{ req/s}$$


2. **Input Token Throughput**：

$$\text{Input Throughput} = \frac{50 \times 512 \text{ tokens}}{10 \text{ s}} = 2560 \text{ input tokens/s}$$


3. **Output Token Throughput**：

$$\text{Output Throughput} = \frac{50 \times 100 \text{ tokens}}{10 \text{ s}} = 500 \text{ output tokens/s}$$



---

## 3. 延迟统计分位数 (Latency Percentiles)

大模型推理具有高动态性（Prompt 长度不同、生成长度不同、KV Cache 动态变化），仅看**平均数（Mean）会严重掩盖少量的长尾卡顿**。因此必须使用百分位数（Percentiles）进行评估：

* **P50（中位数 / Median）**：50% 的请求延迟低于该值。代表**典型用户的正常体验**。
* **P95（95 分位数）**：95% 的请求延迟低于该值。代表**绝大多数用户体验的上限**。
* **P99（99 分位数 / 尾部延迟 Tail Latency）**：99% 的请求延迟低于该值。代表**极端长尾场景**（如长 Prompt 挤占 Batch、显存碎片化触发整理等）。

### 3.1 为什么平均数（Mean）不可靠？

* **示例场景**：测试 100 次请求的 TTFT：
* 其中 99 次请求响应极快，$\text{TTFT} = 20 \text{ ms}$；
* 有 1 次请求因为排队遭遇极端卡顿，$\text{TTFT} = 2000 \text{ ms}$（2 秒）。


* **指标对比**：
* **平均数 (Mean)**：$\frac{99 \times 20 + 1 \times 2000}{100} = 39.8 \text{ ms}$（看起来依然非常优秀）；
* **P50**：$20 \text{ ms}$；
* **P99**：$2000 \text{ ms}$（真实暴露了第 100 个用户遭到的严重体验损坏）。



---

## 4. 总结：性能日志的标准格式示例

一个合格的 AI Infra 性能评测日志或报告输出应包含如下要素：

```text
==================== LLM Benchmark Summary ====================
[Environment] Model: Qwen2.5-1.5B | Precision: BF16 | Device: NVIDIA RTX 4090
[Workload]    Batch Size: 8 | Prompt Len: 512 | Gen Len: 64 | Requests: 100

[Latency Metrics]
  - TTFT (ms)  : P50 = 85.2 | P95 = 112.0 | P99 = 145.8 | Mean = 88.4
  - TPOT (ms)  : P50 = 18.1 | P95 =  20.4 | P99 =  24.1 | Mean = 18.5

[Throughput Metrics]
  - Request Throughput     : 12.4 req/s
  - Input Token Throughput : 6348.8 input tokens/s
  - Output Token Throughput: 793.6 output tokens/s
===============================================================

```