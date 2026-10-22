# Profiling 实战：可信计时、trace 与 Roofline

## 1. 计时区间先定清楚

基准测试要回答的是「重复执行同一个算子，纯 GPU 上花多少时间」。因此输入输出必须提前分配到 GPU，显存分配、数据初始化、预热与答案核对全部排除在计时区间之外。

这样做的目的很直接：把注意力集中在计算与访存本身，而不是被单次创建数组的固定开销干扰。顺序上先固定输入、用 `empty_like` 预分配输出，再进预热与采样循环：

```python
c = torch.empty_like(a)          # 只分配，不填充
for _ in range(warmup):          # 先执行，再开始收集
    torch.add(a, b, out=c)

times = []
for _ in range(repeat):
    start.record()
    torch.add(a, b, out=c)
    end.record()
    end.synchronize()
    times.append(start.elapsed_time(end))
```

`torch.empty_like` 只在显存上划出一块空间，不写数据；配合 `out=` 参数，每次迭代都写入同一地址，避免在循环体内反复申请显存。

## 2. 从毫秒到有效带宽

只看毫秒数，不同规模、不同算子之间无法比较。换算成有效带宽就有了共同尺度：一次向量加法读写 `3N` 个 FP32 元素，即 `12N` 字节。

$$
BW_{\mathrm{effective}} = \frac{12N}{t}
$$

以 `N = 16 777 216`、中位耗时 0.338 ms 为例，有效带宽约 595 GB/s。这个数字的含义是「单位时间内实际搬运了多少有效字节」，它与硬件标称带宽不是一回事：标称值描述物理上限，有效带宽描述这份实现实际达成的搬运速率。

统计口径同样要固定。预热 20 次、采样 200 次是常见配置；报告至少给出 `min` / `median` / `mean` 三个值。设备未锁频、后台有其它 GPU 负载时，绝对数值只用于同一台机器上的相对比较。

## 3. rocprofv3：把一次启动摊开看

GPU event 只能回答「这段区间花了多少时间」，要看「每一次 kernel 启动的细节」，得用 kernel trace。`rocprofv3` 来自同一个 ROCm SDK 环境，通常这样挂接：

```bash
rocprofv3 --kernel-trace -o logs/kt_coalesced.csv -f csv \
    -- ./vector_add_bench --kernel coalesced --size 16777216 --block 256 \
       --warmup 5 --repeat 10
```

`--` 是关键：左侧是 profiler 参数（`--kernel-trace` 只记内核调用，`-f csv` 指定格式，`-o` 指定前缀），右侧是被分析的程序。运行后会生成 `..._kernel_trace.csv`，同目录可能附带 `_agent_info.csv`（GPU 拓扑信息，不是执行流水）。

### 3.1 先选对记录

一个容易踩的坑：文件里往往不止目标 kernel。以本例为例，16 条 Dispatch 记录中只有 15 条属于向量加法（5 次预热 + 10 次正式），最后一条是运行时回收数据触发的 `__amd_rocclr_copyBuffer`。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/hellogpu_6_1.svg" width="80%" alt="trace 记录分组：预热、正式执行与运行时拷贝" />
  <p>图 1：trace 记录分组，统计前先按内核名筛选，再排除预热</p>
</div>

所以过滤必须做两步：先按 `Kernel_Name` 精确筛选，再按时序剔除预热样本。直接全选表格掐头去尾，会把底层拷贝内核算进均值。

### 3.2 从字段算真实耗时

一条稳态记录的字段含义：

| 字段 | 示例 | 含义 |
|---|---|---|
| `Kernel_Name` | `kernel_coalesced(float const*, ...)` | 内核符号与参数类型 |
| `Dispatch_Id` | 6 | 启动批次编号 |
| `Start_Timestamp` | 62486600995446 | GPU 开始执行的硬件时间戳（ns） |
| `End_Timestamp` | 62486601325050 | 执行完毕时间戳（ns） |
| `Workgroup_Size_X` | 256 | 工作组（Block）线程宽度 |
| `Grid_Size_X` | 16777216 | 网格全局线程总数，不是 Block 数 |

时间戳的绝对值没有独立意义，相减才有效：`329604 ns ≈ 0.330 ms`。这个数不包含 CPU 发起调用的调度开销，也不包含驱动排队，是 kernel 在 GPU 上的净耗时。

还有两条纪律：**Profiling 运行不能替代基准测试**（插桩本身有开销），两者必须分开；**CSV 里 `Grid_Size_X` 是线程总数**，换算 block 数要再除以 `Workgroup_Size_X`。

## 4. Roofline：一次加法落在屋顶的什么位置

Roofline 把计算量、数据量与时间放在同一张图上。先算算法的工作量：处理 `N` 个元素需要 `N` 次浮点加法，读写 `3N` 个 FP32 元素。

$$
F = N\ \text{FLOP},\qquad Q = 3N \times 4 = 12N\ \text{Byte}
$$

$$
AI = \frac{F}{Q} = \frac{1}{12} \approx 0.0833\ \text{FLOP/Byte}
$$

这个比值叫**算术强度（Arithmetic Intensity）**，含义是每搬运 1 字节数据能完成多少次浮点运算。

两条约束分别给出斜线与水平线：

$$
P \le BW \times AI,\qquad P \le P_{\mathrm{compute}}
$$

$$
P_{\mathrm{roof}} = \min(BW \times AI,\ P_{\mathrm{compute}})
$$

斜线来自带宽（AI 越大，同样数据能支撑的计算越多），水平线来自算力上限，两者拼起来像屋顶，交点叫**拐点**：

$$
AI_{\mathrm{knee}} = \frac{P_{\mathrm{compute}}}{BW}
$$

拐点左侧说明带宽是更紧的约束，优先检查数据访问与复用；右侧说明算力更紧，优先检查计算路径与指令效率。

### 4.1 用实测参考线代入

一组可用的参考值来自两个独立工作负载：FP32 `copy_` 实测约 535 GB/s，FP32 矩阵乘约 10.6 TFLOPS，对应拐点约 19.9 FLOP/Byte（两者都是当时达到的性能，不是硬件峰值）。

把 `N = 16 777 216` 的向量加法代进去：`F = 16 777 216` FLOP，`Q = 192 MiB`，历史最短耗时 0.3338 ms，于是

$$
P = \frac{F}{t} \approx 50.3\ \text{GFLOPS},\qquad BW_{\mathrm{effective}} = \frac{Q}{t} \approx 603\ \text{GB/s}
$$

同样工作量下，跨步访存版（`stride=32`，2.25 ms）只有 89.7 GB/s。两个工作点的横坐标完全相同（AI 一样），纵坐标差 6.73 倍——这正是合并访存被破坏的代价。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/hellogpu_7_1.png" width="82%" alt="向量加法两个工作点与实测参考线" />
  <p>图 2：两个工作点与 copy、GEMM 实测参考线</p>
</div>

### 4.2 读图的三条纪律

1. **先看坐标轴**：双对数刻度下，相同距离代表相同倍数关系，不是绝对差值；
2. **再看横坐标**：工作点在拐点左侧，说明「先查数据访问」是合理起点，但不能证明已跑满带宽；
3. **最后看落差**：图中能确认的只有「完成相同逻辑工作量，某个版本花了更多时间」；显存事务、线程内循环、并行度各占多少，还需进一步实验。

有效带宽高于 `copy_` 参考值并不代表超越硬件：两者逻辑读写量、读写比例、统计量都不同，条件不可直接比。

## 5. 从现象到下一轮实验

Roofline 只负责提出假设，验证还要靠受控对照：

| 观察到 | 可以假设 | 如何区分 |
|---|---|---|
| AI 小且性能低于带宽参考线 | 数据访问方式引入额外开销 | 固定工作划分，只改地址映射 |
| 输入较小、吞吐偏低 | 启动与提交的固定开销占比大 | 增大输入，观察时间与吞吐变化 |
| 多个算子顺序执行 | 中间结果读写或多次启动有代价 | 检查数据依赖，尝试融合 |
| AI 较高但低于算力参考线 | 指令依赖或并行度不足 | 核对工作划分，设计对照 |

记录结论时把「已知事实」与「推论」分开写：环境与输入、计时方法、结果、已有证据、尚未分离的影响、下一步计划。半年后回看这份记录，才分得清哪些是测量，哪些是猜测。

## 写在最后

能测准、能读懂 trace、能把工作点画到屋顶图上，接下来才轮到改代码。下一篇[Element-Wise 与 Reduction](7-ElementWise与Reduction)从两个最基础的算子开始，演示「一轮只改一个设计选择」的优化节奏。
