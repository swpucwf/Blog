# 第一个 HIP 程序与性能计时

## 1. 从向量加法到完整程序

一个可运行的 HIP 程序，外壳永远是六步：分配显存 → 拷入输入 → 启动 kernel → 同步 → 拷回结果 → 校验并释放。变的只是 kernel 内部那几行。

```cpp
#include <hip/hip_runtime.h>

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) c[idx] = a[idx] + b[idx];
}
```

主机侧按这个骨架组织：

```cpp
const int n = 1 << 20;
const size_t bytes = n * sizeof(float);
float *d_a, *d_b, *d_c;
HIP_CHECK(hipMalloc(&d_a, bytes));
HIP_CHECK(hipMalloc(&d_b, bytes));
HIP_CHECK(hipMalloc(&d_c, bytes));
HIP_CHECK(hipMemcpy(d_a, h_a.data(), bytes, hipMemcpyHostToDevice));
HIP_CHECK(hipMemcpy(d_b, h_b.data(), bytes, hipMemcpyHostToDevice));

const int threads = 256;
const int blocks = (n + threads - 1) / threads;
vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
HIP_CHECK(hipGetLastError());          // 启动是否成功
HIP_CHECK(hipDeviceSynchronize());     // 等执行完成

HIP_CHECK(hipMemcpy(h_c.data(), d_c, bytes, hipMemcpyDeviceToHost));
```

几个易错点：

- **块数要向上取整**：`(n + threads - 1) / threads`，否则尾部元素不参与计算；
- **`hipGetLastError()` 与 `hipDeviceSynchronize()` 分工不同**：前者检查启动参数错误（如块大小超限），后者等执行完成并暴露运行期错误；
- **校验放在计时之外**：结果检查要遍历整段数据，混进计时会污染测量。

## 2. 异步执行：为什么不能直接读 CPU 时钟

CPU 发出 GPU 操作后可以继续往下执行，因此「Python 已经跑到下一行」不等于「GPU 已经算完」。直接在提交前后读取 CPU 时钟，量到的往往只是提交耗时，而不是 kernel 执行时间。

正确做法是用 GPU event 在同一条执行流中打两个标记，等结束标记完成后再读时间差：

```python
start.record()
torch.add(a, b, out=c)
end.record()
end.synchronize()
times.append(start.elapsed_time(end))
```

三个 API 的语义要分清：`record()` 把标记放入当前流；`synchronize()` 让主机等到标记完成；`elapsed_time` 返回两点之间的毫秒数。同步之后做的是**读取时间差**，不是这时才开始计时。

<div align="center">
  <img src="https://raw.githubusercontent.com/swpucwf/MyBolgImage/main/img/hellogpu_4_1.svg" width="72%" alt="GPU event 计时区间与同步位置" />
  <p>图 1：event 确定计时区间，同步保证区间已结束</p>
</div>

这个区间是设备侧的一段经过时间，不包含主机与设备间的数据拷贝，也不等于整个程序耗时。对很短的运算，提交节奏与设备状态都可能影响 event 间隔——不能凭单个数字断言「kernel 内部花了多少时间」。

## 3. 预热、重复与三种统计量

正式测量前先跑 5 次作为预热，再重复测量 30 次：

```python
for _ in range(warmup):
    torch.add(a, b, out=c)

times = []
for _ in range(repeat):
    start.record()
    torch.add(a, b, out=c)
    end.record()
    end.synchronize()
    times.append(start.elapsed_time(end))
```

预热的作用是减轻首次执行、频率爬升与状态变化的影响；5 次只是本例配置，不保证所有负载都已稳定。测量结束报告三种统计量：

| 字段 | 含义 | 用法 |
|---|---|---|
| `median_ms` | 排序后的中间值 | 主要比较值，抗少量离群点 |
| `mean_ms` | 平均值 | 与中位数一起看波动 |
| `min_ms` | 最短的一次 | 补充参考，不视为「无干扰真值」 |

只报平均值是最常见的坑：一次系统抖动就能把均值拉高，让人误判某个版本变慢。中位数与最小值同时给出，才看得出「典型性能」和「最好情况」的差距。

## 4. 计时的边界

- **不含 H2D/D2H 拷贝**：event 只覆盖设备侧区间，如果业务关心端到端延迟，要单独测拷贝与前后处理；
- **不含首次初始化**：event 对象本身有延迟初始化成本，提前记录一次可避免它混进第一次测量；
- **不同规模不可直接比**：`--size` 改变数据量，时间自然不同，比较必须在同一规模下进行。

## 5. 一次完整的对照实验

同一段向量加法，CPU 与 GPU 两条路径各跑一遍，输入创建、输出分配和结果检查都放在计时之外，计时内只留加法。这样得到的对比才有意义：

```bash
python benchmark_vector_add.py --size 16777216 --warmup 5 --repeat 30 --cpu-threads 1
```

CPU 侧固定单线程并打印型号与线程数，是为了让结果可解释——CPU 一栏究竟用了什么配置，必须写在结果旁边。

## 写在最后

能把程序跑通、把时间量准，才具备讨论「快不快」的资格。下一篇[Profiling 实战](6-Profiling实战)把测量推进一层：可信计时、kernel trace 与 Roofline，回答「慢在哪一段」。
