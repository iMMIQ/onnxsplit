# ONNX模型切分工具设计文档

## 1. 项目概述

### 1.1 目标
设计一个ONNX模型切分工具，通过算子复制方式将大模型切分为多个可顺序执行的部分，降低单设备内存峰值。

### 1.2 核心功能
- 读取ONNX模型文件
- 根据配置文件或命令行参数切分算子
- 智能识别可切分轴
- 自动重连数据流
- 支持内存限制自动调整切分数

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      CLI 入口层                              │
│  解析命令行参数，协调各模块执行                               │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                   配置管理模块                               │
│  - 加载配置文件（YAML）                                      │
│  - 合并命令行默认值                                          │
│  - 验证配置有效性                                            │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                   ONNX模型分析模块                           │
│  - 解析ONNX图结构                                            │
│  - 构建算子依赖关系图                                        │
│  - 推断所有中间张量的shape                                   │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                   切分方案生成模块                           │
│  - 分析每个算子的可切分轴                                    │
│  - 根据配置和内存限制确定切分数                              │
│  - 生成算子复制方案                                          │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                   图变换执行模块                             │
│  - 复制算子节点                                              │
│  - 插入Slice/Concat算子                                      │
│  - 重新连接数据流                                            │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                   形状推理与验证模块                         │
│  - 运行ONNX shape inference                                 │
│  - 验证图结构有效性                                          │
│  - 生成内存占用报告                                          │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                   输出模块                                   │
│  - 保存切分后的ONNX模型                                     │
│  - 生成切分报告（JSON）                                     │
└─────────────────────────────────────────────────────────────┘
```

## 3. 配置文件格式

```yaml
# onnxsplit配置文件

# 全局默认设置
global:
  default_parts: 1
  max_memory_mb: null

# 算子级别的精确配置
operators:
  "/model/Conv_0":
    parts: 4
    axis: 0               # 可选：指定切分轴

  "/model/MatMul_*":      # 通配符匹配
    parts: 2

# 切分轴规则的智能配置
axis_rules:
  - op_type: "Conv"
    prefer_axis: 0        # Batch维度

  - op_type: "MatMul"
    prefer_axis: "batch"

  - op_type: "LayerNorm"
    prefer_axis: null     # 不可切分

# 内存限制规则
memory_rules:
  auto_adjust: true
  overflow_strategy: "binary_split"
```

**配置优先级**：算子实例配置 > 算子类型配置 > 全局配置 > 命令行参数

## 4. 切分轴识别逻辑

| 算子类型 | 可切分轴 | 说明 |
|---------|---------|------|
| Element-wise (Add, Mul, Relu, etc.) | 任意轴 | 各元素独立计算 |
| Conv | Axis 0 (Batch) | 权重共享于batch |
| MatMul | Batch轴, 序列长度轴 | 需分析shape |
| Reduce (ReduceMean, ReduceSum, etc.) | 仅非归约轴 | 不能沿被归约的轴切分 |
| Reshape/Gather/Scatter | 需特殊分析 | 取决于具体操作 |
| LayerNorm/BatchNorm | Axis 0 (Batch) | 统计量沿特定轴计算 |

## 5. 数据流重连算法

### 5.1 重连策略
```
原始图:    A  →  B  →  C
           ↓      ↓      ↓
切分后:   A0 →  B0 →  C0
          A1 →  B1 →  C1
          A2 →  B2 →  C2
```

### 5.2 切分数匹配情况分类

当相邻算子切分数不同时，需要特殊处理：

| 情况 | parts(A) | parts(B) | 处理方式 |
|------|----------|----------|----------|
| 相同 | N | N | 直接1对1连接 |
| 整除 | N | M×N | A每份Split成M份 → B |
| 整除 | M×N | N | A每M份Concat → B每份 |
| 任意 | M | N (无整除关系) | Slice+Concat复杂重排 |

### 5.3 任意切分数的复杂重排

当 `parts(A)` 和 `parts(B)` 无整除关系时，使用 Slice + Concat 进行数据重排。

**示例：parts(A)=3, parts(B)=2**

假设原始 batch = 6：
```
原始数据: [0, 1, 2, 3, 4, 5]

A切3份:           B切2份:
┌─────────┐        ┌─────────────┐
│ A0: [0,1]│   ?    │ B0: [0,1,2] │
├─────────┤   →    ├─────────────┤
│ A1: [2,3]│   ?    │ B1: [3,4,5] │
├─────────┤   →    └─────────────┘
│ A2: [4,5]│   ?
└─────────┘

问题：B0需要的[0,1,2]跨越了A0和A1的边界
```

**解决方案可视化**：
```
A0: [0,1] ──┐
           ├── Slice([0,1]) ──┐
A1: [2,3] ──┤                ├── Concat ─→ B0: [0,1,2]
           ├── Slice([2]) ───┘
A2: [4,5] ──┘

A0: [0,1] ──┐
           ├── Slice([3]) ───┐
A1: [2,3] ──┤                ├── Concat ─→ B1: [3,4,5]
           ├── Slice([4,5]) ─┘
A2: [4,5] ──┘
```

### 5.4 重连算法实现

```python
def reconnect_with_adaptation(
    src_op: str,
    dst_op: str,
    src_parts: int,
    dst_parts: int,
    batch_size: int
) -> List[NodeProto]:
    """处理任意切分数组合的数据重连"""

    if src_parts == dst_parts:
        # 情况1: 切分数相同，直接连接
        for i in range(src_parts):
            connect(f"{src_op}_{i}", f"{dst_op}_{i}")
        return []

    # 计算每份大小
    src_chunk = batch_size // src_parts
    dst_chunk = batch_size // dst_parts

    nodes = []

    # 为每个目标算子构建输入
    for dst_i in range(dst_parts):
        dst_start = dst_i * dst_chunk
        dst_end = (dst_i + 1) * dst_chunk

        slice_outputs = []

        # 找出所有与目标区间重叠的源切片
        for src_i in range(src_parts):
            src_start = src_i * src_chunk
            src_end = (src_i + 1) * src_chunk

            # 计算重叠区间
            overlap_start = max(dst_start, src_start)
            overlap_end = min(dst_end, src_end)

            if overlap_start < overlap_end:
                # 从源切片中提取需要的部分
                local_start = overlap_start - src_start
                local_end = overlap_end - src_start

                slice_node = make_node(
                    "Slice",
                    inputs=[
                        f"{src_op}_{src_i}",
                        make_tensor('starts', INT64, [0], [local_start]),
                        make_tensor('ends', INT64, [0], [local_end]),
                        make_tensor('axes', INT64, [0], [0])
                    ],
                    outputs=[f"slice_{src_op}_{src_i}_for_{dst_op}_{dst_i}"]
                )
                nodes.append(slice_node)
                slice_outputs.append(slice_node.output[0])

        # 将多个切片合并为一个目标输入
        if len(slice_outputs) > 1:
            concat_node = make_node(
                "Concat",
                inputs=slice_outputs,
                outputs=[f"concat_to_{dst_op}_{dst_i}"],
                axis=0
            )
            nodes.append(concat_node)
            connect(f"concat_to_{dst_op}_{dst_i}", f"{dst_op}_{dst_i}")
        else:
            connect(slice_outputs[0], f"{dst_op}_{dst_i}")

    return nodes
```

### 5.5 算法步骤总结
1. **输入切分**：插入 Split 算子沿指定轴切分输入
2. **算子复制**：将需要切分的算子复制N份
3. **数据流重连**：
   - 相同切分数：直接1对1连接
   - 整除关系：使用 Split 或 Concat
   - 任意组合：使用 Slice + Concat 复杂重排
   - 后续算子不切分：插入 Concat 合并所有输入
4. **输出合并**：最终输出插入 Concat 合并

## 6. 内存估算与自动切分

### 6.1 内存估算
```python
def estimate_tensor_memory(tensor: TensorProto) -> int:
    dtype_size = {
        TensorProto.FLOAT: 4,
        TensorProto.FLOAT16: 2,
        TensorProto.INT32: 4,
        # ...
    }.get(tensor.data_type, 4)
    return prod(tensor.shape) * dtype_size
```

### 6.2 自动切分策略
当内存超过限制时，使用二分查找确定最优切分数：
- 假设沿batch轴切分，内存近似线性减少
- 上限256份，超过64份时发出警告

## 7. 命令行接口

```bash
onnxsplit model.onnx --config config.yaml \
          --parts 2 \              # 全局默认切分数
          --max-memory 512 \       # 单份内存上限(MB)
          --output model_split.onnx \
          --report report.json     # 生成报告
```

## 8. 项目目录结构

```
onnxsplit/
├── src/onnxsplit/
│   ├── __init__.py
│   ├── __main__.py           # CLI入口
│   ├── cli/                  # 命令行接口
│   ├── config/               # 配置管理
│   ├── analyzer/             # ONNX模型分析
│   ├── splitter/             # 切分逻辑核心
│   ├── memory/               # 内存分析
│   ├── transform/            # 图变换
│   └── utils/
├── tests/
│   └── fixtures/models/
└── docs/
    └── design.md
```

## 9. 核心数据结构

```python
@dataclass
class SplitPlan:
    """切分方案"""
    operator_name: str
    parts: int
    axis: int | None
    slice_ranges: list[tuple[int, int]]

@dataclass
class TensorMetadata:
    """张量元数据"""
    name: str
    shape: tuple[int, ...]
    dtype: onnx.TensorProto.DataType
    memory_bytes: int

@dataclass
class OperatorInfo:
    """算子信息"""
    name: str
    op_type: str
    attributes: dict
    splitable_axes: set[int]
    input_tensors: list[TensorMetadata]
    output_tensors: list[TensorMetadata]
```

## 10. 输出报告格式

```json
{
  "model": "model_split.onnx",
  "memory_analysis": {
    "total_memory_mb": 1024,
    "peak_memory_mb": 256,
    "split_statistics": {
      "total_operators": 100,
      "split_operators": 15,
      "unsplit_operators": 85
    },
    "operator_details": [
      {
        "name": "/model/Conv_0",
        "original_memory_mb": 512,
        "split_parts": 4,
        "per_part_memory_mb": 128,
        "split_axis": 0
      }
    ]
  }
}
```

## 11. 参考资源

- [ONNX Python API](https://onnx.ai/onnx/api/)
- [onnx.helper Module](https://onnx.ai/onnx/api/helper.html)
- [ONNX Shape Inference](https://onnx.ai/onnx/repo-docs/ShapeInference.html)
- [Creating and Modifying ONNX Model](https://leimao.github.io/blog/ONNX-Python-API/)
- [ONNX Split Operator](https://onnx.ai/onnx/operators/onnx__Split.html)
