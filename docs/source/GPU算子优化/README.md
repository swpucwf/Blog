# GPU 算子优化

> 内容整理自开源教程 [hello-gpu](https://github.com/datawhalechina/hello-gpu)，作为参考笔记。

以 AMD Radeon RX 9070 XT + ROCm 10.0 为实验平台，沿「理解硬件 → profiling 取证 → 手写算子优化 → Agent 自动化」的主线，把向量加法、归约、Softmax、GEMM、Attention、RMSNorm 六类算子从 naive 版本优化到接近硬件极限，最后把整条流程交给一个能调用工具的 Agent。

## 目录

- [1-从硬件到算子的优化主线](/source/GPU算子优化/1-从硬件到算子的优化主线)
- [2-环境准备与三道门](/source/GPU算子优化/2-环境准备与三道门)
- [3-编程模型与wavefront执行](/source/GPU算子优化/3-编程模型与wavefront执行)
- [4-数据存储与访问](/source/GPU算子优化/4-数据存储与访问)
- [5-第一个HIP程序与性能计时](/source/GPU算子优化/5-第一个HIP程序与性能计时)
- [6-Profiling实战](/source/GPU算子优化/6-Profiling实战)
- [7-ElementWise与Reduction](/source/GPU算子优化/7-ElementWise与Reduction)
- [8-Softmax与GEMM](/source/GPU算子优化/8-Softmax与GEMM)
- [9-Attention与FusedRMSNorm](/source/GPU算子优化/9-Attention与FusedRMSNorm)
- [10-算子优化Agent](/source/GPU算子优化/10-算子优化Agent)
