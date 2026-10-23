# 重排：MMR 与 DPP

## 1. 为什么需要重排

精排按 CTR 降序输出，头部往往高度同质：连续几个同品类商品、同风格视频。后果有两个：用户审美疲劳，长尾内容没有曝光。

重排的任务是在保留相关性的前提下，调节整张列表的多样性与新颖性。两种经典做法都基于贪心，但度量方式不同。

## 2. MMR：边际相关

MMR 定义一个边际收益函数，每轮选一个「相关性高、且与已选内容不太像」的物品：

$$
\mathrm{MR}(i) = \lambda \cdot \mathrm{Rel}(i) - (1-\lambda)\cdot \max_{j\in S}\mathrm{Sim}(i,j)
$$

- \(\mathrm{Rel}(i)\)：直接继承精排分数；
- \(\max_j \mathrm{Sim}(i,j)\)：与已选集合中最相似那个的相似度，作为多样性惩罚；
- \(\lambda\)：权衡参数。趋近 1 退化为纯精排，趋近 0 强制多样性优先。

候选多时，逐对计算相似度开销大，可以只与**最近选择的 w 个物品**比较（滑动窗口）：

$$
\mathrm{MR}_{\text{win}}(i) = \lambda\cdot\mathrm{Rel}(i) - (1-\lambda)\cdot\max_{j\in W}\mathrm{Sim}(i,j)
$$

### 2.1 一个手算例子

5 个候选，精排分为 A 0.95、B 0.90、C 0.85、D 0.80、E 0.75；相似度矩阵中 A-C 高达 0.8、B-D 0.7。取 \(\lambda=0.7\)：

**第一轮**：直接选 A（相关性最高）。

**第二轮**：

| 候选 | 计算 | 得分 |
|---|---|---|
| B | 0.90 − 0.7 × 0.2 | 0.76 |
| C | 0.85 − 0.7 × 0.8 | 0.29 |
| D | 0.80 − 0.7 × 0.1 | 0.73 |
| E | 0.75 − 0.7 × 0.3 | 0.54 |

选 B。

**第三轮**（对比 S=[A,B]）：

| 候选 | 计算 | 得分 |
|---|---|---|
| C | 0.85 − 0.7 × max(0.8, 0.1) | 0.29 |
| D | 0.80 − 0.7 × max(0.1, 0.7) | 0.31 |
| E | 0.75 − 0.7 × max(0.3, 0.4) | 0.47 |

选 E。最终 `[A, B, E]`，而精排是 `[A, B, C]`——C 因为与 A 太像被压到后面，这就是 MMR 的作用。

### 2.2 实现要点

```python
S = [max(candidates, key=lambda x: x.rel)]      # 先选最高分
while len(S) < k and candidates:
    window = S[-window_size:] if window_size else S
    best = max(candidates, key=lambda x: lam * x.rel
               - (1 - lam) * max(sim(x, s) for s in window))
    S.append(best); candidates.remove(best)
```

两个工程细节：**必须用深拷贝**，否则删候选会污染上游列表；**相似度函数要可插拔**，稠密向量用余弦，稀疏特征用类目重合度，选择哪种直接决定重排效果。

## 3. DPP：用行列式度量多样性

MMR 只看「与已选最相似的那一个」，无法刻画三个以上物品之间的冗余。行列式能表达这种整体关系。

### 3.1 行列式为什么能表示多样性

把物品表示为向量 \(x_i\)，相似度矩阵 \(S = X^TX\)。行列式的几何意义是列向量张成的超体积：向量越线性相关（越像），体积越小；越正交（越不像），体积越大。

两个例子最直观：

$$
|S_{a,b}| = \begin{vmatrix}1 & 0.9\\ 0.9 & 1\end{vmatrix} = 0.19,\qquad
|S_{b,d}| = \begin{vmatrix}1 & 0.1\\ 0.1 & 1\end{vmatrix} = 0.81
$$

相似度 0.9 的一对行列式只有 0.19，相似度 0.1 的则有 0.81。**行列式大 = 多样性高**。

### 3.2 把相关性揉进核矩阵

只优化多样性会推出不相关的内容，于是构造核矩阵：

$$
L = \mathrm{Diag}(r)\cdot S\cdot \mathrm{Diag}(r),\qquad L_{ij} = r_i r_j S_{ij}
$$

其中 \(r_i\) 是精排相关性分数。取子集 \(R_u\) 的行列式：

$$
|L_{R_u}| = \prod_{i\in R_u} r_{u,i}^2 \cdot |S_{R_u}|
$$

取对数后变成两项相加：

$$
\log|L_{R_u}| = \sum_{i\in R_u}\log r_{u,i}^2 + \log|S_{R_u}|
$$

第一项管相关性，第二项管多样性——这和 MMR 的线性组合本质一致，只是多样性从「两两最大相似」升级成「整体体积」。实践中再加一个 \(\theta\) 调节两者权重。

### 3.3 贪心求解：为什么只需看 \(d_i^2\)

直接求 DPP 的最大后验子集很难，Hulu 的改进贪心把它化简成逐步选择：

$$
j = \arg\max_{i \in Z\setminus Y_g}\ \log\det(L_{Y_g\cup\{i\}}) - \log\det(L_{Y_g})
$$

对半正定矩阵做 Cholesky 分解 \(L_{Y_g}=VV^T\)，加入新物品后分块分解可推出：

$$
\det(L_{Y_g\cup\{i\}}) = \det(L_{Y_g})\cdot d_i^2
$$

于是边际增益只与 \(d_i^2\) 有关，选择规则简化为 \(j=\arg\max_i \log d_i^2\)。每选一个物品，其余候选的 \(c_i\)、\(d_i^2\) 可以增量更新，无需重算整个行列式：

$$
e_i = \frac{L_{ji} - \langle c_j, c_i\rangle}{d_j},\qquad
c_i \leftarrow [c_i,\ e_i],\qquad
d_i^2 \leftarrow d_i^2 - e_i^2
$$

### 3.4 实现

```python
di2s = np.diag(kernel)                  # 初始化 d_i^2
selected = [np.argmax(di2s)]            # 第一件
while len(selected) < k:
    j = selected[-1]
    ci = cis[:len(selected)-1, j]; dj = math.sqrt(di2s[j])
    e = (kernel[j, :] - ci @ cis[:len(selected)-1, :]) / dj
    cis[len(selected)-1, :] = e
    di2s -= e ** 2
    nxt = np.argmax(di2s)
    if di2s[nxt] < 1e-10: break         # 数值保护
    selected.append(nxt)
```

`epsilon` 不是可选项：浮点误差会让 \(d_i^2\) 出现负值，没有保护会直接选到无效项。

## 4. MMR 与 DPP 对比

| 维度 | MMR | DPP |
|---|---|---|
| 多样性度量 | 与已选最大相似度 | 子集行列式（整体体积） |
| 计算复杂度 | \(O(k \cdot n \cdot w)\) | \(O(k \cdot n^2)\)，可增量更新 |
| 调参 | \(\lambda\) 直观 | \(\theta\) 与核矩阵构造 |
| 表达力 | 只看单点最相似 | 能表达多物品冗余 |
| 适用 | 快速上线、规则可控 | 列表质量要求高、候选规模适中 |

一个实用的组合：**先用 MMR 做粗排多样性，再对 Top-20 用 DPP 精调**。MMR 便宜，DPP 贵但只在小区间上跑，成本可控。

## 5. 落地注意

- **相似度定义决定一切**：用类目、向量还是标签，效果差异远大于换算法；
- **列表长度与窗口**：窗口太小退化成局部最优，太大则计算爆掉，常用 5～10；
- **不要与业务规则打架**：广告位、运营坑位是硬约束，应在重排后再插入，而不是让算法「学会」留位；
- **线上看多样性与消费深度**：重排的收益不体现在 CTR 上，更多在人均消费条数、次日留存这些指标。

## 写在最后

MMR 与 DPP 都是「启发式 + 贪心」，不需要训练，代价是无法针对用户个性化。下一篇看个性化重排：把用户偏好直接建模进列表级打分，用 Transformer 或排列组合模型端到端生成整张列表。
