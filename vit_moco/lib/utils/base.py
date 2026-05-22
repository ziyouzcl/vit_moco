"""
base — 训练随机种子与通用小工具

本模块提供训练入口常用的可复现性设置：在 ``finetuning_main`` 解析配置后调用 ``set_seed``，
统一固定 Python / PyTorch 随机性，并按是否指定 seed 切换 cuDNN 的 deterministic 与 benchmark 模式。
另导出 ``partial`` 包装，便于在配置或工厂函数中做偏函数绑定。

主要函数
-------
partial
    对 ``functools.partial`` 的薄封装，签名与标准库一致，供 ``__all__`` 与配置侧统一从 ``fmlct.lib.utils`` 导入。

set_seed
    ``seed`` 非空时：设置 ``random``、``torch`` 种子，并 ``cudnn.deterministic=True``（更可复现、可能更慢）；
    ``seed`` 为 ``None`` 时：``cudnn.benchmark=True``，由 cuDNN 自动选较快卷积算法。

使用场景
--------
- ``finetuning_main`` 启动时根据 YAML 中的 ``seed`` 调用，保证实验可重复。
- 不固定 seed 的正式训练跑分，传入 ``seed=None`` 以启用 benchmark 加速。
"""
import torch
import torch.distributed as dist  # 预留：与其它 utils 模块风格一致，当前本文件未使用
import torch.backends.cudnn as cudnn
import random
import warnings
from functools import partial as _partial

__all__ = [
    "partial",
    "set_seed",
]


def partial(func, *args, **kwargs):
    """返回 functools.partial(func, *args, **kwargs)，用于延迟绑定默认参数。"""
    return _partial(func, *args, **kwargs)


def set_seed(seed=None):
    """
    设置训练随机种子与 cuDNN 行为。

    seed 有值：固定 random / torch，开启 deterministic（并发出性能警告）。
    seed 为 None：不固定种子，开启 cudnn.benchmark 以追求速度。
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        # np.random.seed(seed)  # 若 DataLoader 使用 numpy 随机，可取消注释
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    else:
        cudnn.benchmark = True
