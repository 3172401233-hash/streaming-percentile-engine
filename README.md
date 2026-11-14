# streaming-percentile-engine

Python 流式分位数引擎，处理加权样本、分片合并、有界内存和确定性误差报告。
适用于统计接口在批处理阶段计算 p50 / p90 / p99：原始样本不落地，只保留
随压缩参数增长的概要（sketch），多个分区的概要可在汇总阶段合并。

纯标准库实现，无第三方依赖，Python 3.9+。

## 快速开始

```python
from streaming_percentile import QuantileEngine

engine = QuantileEngine(compression=100)   # 压缩参数：越大越准、占用越多
engine.add(12.5, weight=1.0, partition="shard-0")
engine.add(7.25, weight=2.0, partition="shard-1")

# 汇总阶段：合并另一个引擎的所有分片
engine.merge(other_engine)

result = engine.quantile(0.99)             # 全部分区合并后的 p99
print(result.value)                        # 估计的分位数值
print(result.rank_error)                   # 候选排名误差（权重单位）
print(result.total_weight)                 # 实际摄入的总权重

per_shard = engine.quantile(0.5, partition="shard-0")  # 单分区查询
```

## 设计

- 底层是加权 t-digest 变体：样本先进缓冲区，缓冲区满后排序并与已有
  质心（centroid）按 k1 尺度函数重新聚类——靠近中位数的簇可以大，
  靠近尾部（p99 附近）的簇必须小，因此尾部精度高。
- **内存有界**：保留的质心数量约为 `compression / 2`，与样本总数无关；
  缓冲区容量同样是 `O(compression)`。
- **确定性**：全程只有排序（稳定排序），没有任何随机数。相同输入序列
  在任何一次运行中产生完全相同的结果。
- **可合并**：`merge()` 把另一个 sketch 的质心作为加权点重新摄入，
  分片可以在批处理阶段独立构建、在汇总阶段任意合并。

## 误差说明（重要）

`QuantileResult.rank_error` 是**候选排名误差**，以权重为单位：返回值
在完整加权样本中的真实排名预期落在 `q * total_weight ± rank_error`
区间内。该上界由 sketch 的簇结构推导（插值涉及的两个簇所覆盖的排名
区间的一半），`relative_rank_error` 属性给出其占总权重的比例。

请注意：这是一个**近似保证，而非精确承诺**。t-digest 的误差界来自
聚类结构的启发式推导，在常见分布上实测误差通常远小于声明值，但它
不是对每一次查询都严格成立的数学上界。压缩参数越大，声明误差越小
（大致与 `compression` 成反比）。如果需要严格保证，请对小规模数据
使用精确排序。

## 边界行为

| 输入 | 行为 |
| --- | --- |
| NaN / ±inf 的值 | 抛出 `ValueError` |
| 负权重、NaN、无穷权重 | 抛出 `ValueError` |
| 零权重 | 样本被忽略，不计入总权重 |
| 空分区 / 未注册的分区 | 返回 `QuantileResult(value=None, rank_error=0.0, total_weight=0.0)`，不抛异常 |
| 极端重复值（常数流） | 正常合并为少量质心，返回精确值 |
| `q` 不在 `[0, 1]` | 抛出 `ValueError` |
| `compression < 1` 或非有限值 | 抛出 `ValueError` |

## 测试

```bash
python -m unittest
```

测试覆盖均匀、长尾、偏斜、常数、负数和大量重复值分布，对照小规模
精确排序验证返回值的排名误差不超过引擎自身声明的候选误差；并验证
不同插入顺序、分片合并（含不同合并顺序）下结果均在声明误差内，
以及质心数量只随压缩参数增长、不随样本数增长。
