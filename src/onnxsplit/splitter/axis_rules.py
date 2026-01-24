"""切分轴识别规则"""
from dataclasses import dataclass

from onnxsplit.analyzer.operator import OperatorInfo


@dataclass(frozen=True)
class SplitableAxes:
    """可切分轴集合

    Attributes:
        axes: 可切分的轴索引集合
        reason: 原因说明
    """
    axes: set[int]
    reason: str

    @classmethod
    def empty(cls, reason: str = "No splitable axes") -> "SplitableAxes":
        """创建空的可切分轴集合"""
        return cls(set(), reason)

    @classmethod
    def single(cls, axis: int, reason: str = "") -> "SplitableAxes":
        """创建单轴可切分集合"""
        return cls({axis}, reason)

    def __contains__(self, axis: int) -> bool:
        """检查轴是否可切分"""
        return axis in self.axes

    def __len__(self) -> int:
        """可切分轴数量"""
        return len(self.axes)

    def __repr__(self) -> str:
        return f"SplitableAxes(axes={self.axes}, reason={self.reason!r})"


class AxisAnalyzer:
    """切分轴分析器

    基于算子类型和属性，智能识别可以切分的轴。
    """

    # Element-wise算子类型（输入输出形状相同，各元素独立计算）
    ELEMENT_WISE_OPS = {
        "Add",
        "Sub",
        "Mul",
        "Div",
        "Pow",
        "Sqrt",
        "Relu",
        "LeakyRelu",
        "PRelu",
        "Sigmoid",
        "Tanh",
        "Softplus",
        "Elu",
        "Abs",
        "Neg",
        "Exp",
        "Log",
        "Sin",
        "Cos",
        "Min",
        "Max",
        "Clip",
        "Floor",
        "Ceil",
        "Round",
        "Not",
        "And",
        "Or",
        "Xor",
        "Equal",
        "Greater",
        "Less",
        "Cast",
        "Identity",
    }

    def __init__(self):
        """初始化分析器"""
        pass

    def analyze(self, op_info: OperatorInfo) -> SplitableAxes:
        """分析算子的可切分轴

        Args:
            op_info: 算子信息

        Returns:
            可切分轴集合
        """
        op_type = op_info.op_type

        # 获取输入形状（使用第一个输入）
        if not op_info.input_tensors:
            return SplitableAxes.empty("No input tensors")

        input_shape = op_info.input_tensors[0].shape
        if not input_shape:
            return SplitableAxes.empty("Empty input shape")

        # 根据算子类型分析
        if op_type in self.ELEMENT_WISE_OPS:
            return self._analyze_elementwise(op_info, input_shape)
        elif op_type == "Conv":
            return self._analyze_conv(op_info, input_shape)
        elif op_type == "MatMul":
            return self._analyze_matmul(op_info)
        elif op_type.startswith("Reduce"):
            return self._analyze_reduce(op_info)
        elif op_type == "BatchNormalization":
            return self._analyze_batch_norm(op_info, input_shape)
        elif op_type == "LayerNormalization":
            return self._analyze_layer_norm(op_info, input_shape)
        elif op_type == "Softmax":
            return self._analyze_softmax(op_info, input_shape)
        elif op_type in ("MaxPool", "AveragePool", "GlobalAveragePool", "GlobalMaxPool"):
            return self._analyze_pooling(op_info, input_shape)
        elif op_type == "Flatten":
            return self._analyze_flatten(op_info, input_shape)
        elif op_type == "Transpose":
            return self._analyze_transpose(op_info, input_shape)
        elif op_type in ("Reshape", "Squeeze", "Unsqueeze"):
            return SplitableAxes.empty(f"{op_type} changes tensor structure")
        else:
            # 未知算子保守处理
            return SplitableAxes.empty(f"Unknown operator type: {op_type}")

    def _analyze_elementwise(
        self, op_info: OperatorInfo, input_shape: tuple[int, ...]
    ) -> SplitableAxes:
        """分析Element-wise算子

        Element-wise算子所有输入输出形状相同（广播后），
        各元素独立计算，可以切分任意轴。
        """
        all_axes = set(range(len(input_shape)))
        return SplitableAxes(all_axes, "Element-wise operation")

    def _analyze_conv(
        self, op_info: OperatorInfo, input_shape: tuple[int, ...]
    ) -> SplitableAxes:
        """分析Conv算子

        Conv只能在batch维度(axis=0)切分。
        权重在channel维度共享，不能切分input的channel维度。
        """
        # 只有batch维度可切
        if len(input_shape) >= 1:
            return SplitableAxes.single(0, "Batch dimension for Conv")
        return SplitableAxes.empty("Conv input has no batch dimension")

    def _analyze_matmul(self, op_info: OperatorInfo) -> SplitableAxes:
        """分析MatMul算子

        对于 (B, M, K) @ (B, K, N) = (B, M, N) 的3D情况，
        batch维度可切分。

        对于 (M, K) @ (K, N) = (M, N) 的2D情况，
        通常不可切分（会影响矩阵乘法语义）。
        """
        if not op_info.input_tensors:
            return SplitableAxes.empty("No inputs for MatMul")

        shape_a = op_info.input_tensors[0].shape

        # 3D MatMul有batch维度
        if len(shape_a) == 3:
            return SplitableAxes.single(0, "Batch dimension for 3D MatMul")

        # 2D MatMul不可切
        return SplitableAxes.empty("2D MatMul cannot be split")

    def _analyze_reduce(self, op_info: OperatorInfo) -> SplitableAxes:
        """分析Reduce算子

        只能切分非归约轴。
        """
        if not op_info.input_tensors:
            return SplitableAxes.empty("No inputs for Reduce")

        input_shape = op_info.input_tensors[0].shape
        all_axes = set(range(len(input_shape)))

        # 获取归约轴
        reduce_axes_attr = op_info.get_attribute("axes")
        if reduce_axes_attr is not None:
            reduce_axes = set(reduce_axes_attr)
            # 处理负索引
            normalized_reduce = set()
            for ax in reduce_axes:
                if ax < 0:
                    normalized_reduce.add(len(input_shape) + ax)
                else:
                    normalized_reduce.add(ax)
        else:
            # 默认归约所有轴
            return SplitableAxes.empty("Reduce all axes")

        # 可切分的轴 = 所有轴 - 归约轴
        splitable = all_axes - normalized_reduce
        return SplitableAxes(splitable, f"Non-reduce axes (reducing {normalized_reduce})")

    def _analyze_batch_norm(
        self, op_info: OperatorInfo, input_shape: tuple[int, ...]
    ) -> SplitableAxes:
        """分析BatchNorm算子

        只能切batch维度。
        Channel维度的统计量不可切分。
        """
        if len(input_shape) >= 1:
            return SplitableAxes.single(0, "Batch dimension for BatchNorm")
        return SplitableAxes.empty("BatchNorm input has no batch dimension")

    def _analyze_layer_norm(
        self, op_info: OperatorInfo, input_shape: tuple[int, ...]
    ) -> SplitableAxes:
        """分析LayerNorm算子

        只能切batch维度。
        LayerNorm在最后几维计算统计量。
        """
        if len(input_shape) >= 1:
            return SplitableAxes.single(0, "Batch dimension for LayerNorm")
        return SplitableAxes.empty("LayerNorm input has no batch dimension")

    def _analyze_softmax(
        self, op_info: OperatorInfo, input_shape: tuple[int, ...]
    ) -> SplitableAxes:
        """分析Softmax算子

        只能切分非计算轴。
        """
        if not input_shape:
            return SplitableAxes.empty("Empty input for Softmax")

        # 获取计算轴
        axis_attr = op_info.get_attribute("axis", -1)

        # 标准化轴索引
        if axis_attr < 0:
            axis_attr = len(input_shape) + axis_attr

        all_axes = set(range(len(input_shape)))
        splitable = all_axes - {axis_attr}
        return SplitableAxes(splitable, f"Softmax computed on axis {axis_attr}")

    def _analyze_pooling(
        self, op_info: OperatorInfo, input_shape: tuple[int, ...]
    ) -> SplitableAxes:
        """分析池化算子

        可以切batch维度。
        """
        if len(input_shape) >= 1:
            return SplitableAxes.single(0, "Batch dimension for Pooling")
        return SplitableAxes.empty("Pooling input has no batch dimension")

    def _analyze_flatten(
        self, op_info: OperatorInfo, input_shape: tuple[int, ...]
    ) -> SplitableAxes:
        """分析Flatten算子

        只能切batch维度（在flatten axis之前）。
        """
        # 获取flatten axis
        axis_attr = op_info.get_attribute("axis", 1)

        # flatten axis之前的维度可以切分（通常是batch）
        if axis_attr > 0:
            return SplitableAxes.single(0, "Batch dimension before flatten")
        return SplitableAxes.empty("Flatten from axis 0")

    def _analyze_transpose(
        self, op_info: OperatorInfo, input_shape: tuple[int, ...]
    ) -> SplitableAxes:
        """分析Transpose算子

        如果perm[0] == 0（batch维度不变），可以切分batch。
        """
        perm = op_info.get_attribute("perm")
        if perm and len(perm) > 0 and perm[0] == 0:
            return SplitableAxes.single(0, "Batch dimension preserved in transpose")
        return SplitableAxes.empty("Transpose changes batch dimension")


def get_splitable_axes_for_op(op_info: OperatorInfo) -> SplitableAxes:
    """便捷函数：获取算子的可切分轴

    Args:
        op_info: 算子信息

    Returns:
        可切分轴集合
    """
    analyzer = AxisAnalyzer()
    return analyzer.analyze(op_info)
