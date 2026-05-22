"""
distributed — PyTorch 分布式训练辅助（DDP / MoCo 多卡）

作用
    为多 GPU、多进程训练提供进程组初始化与跨卡张量拼接，供 finetuning_main / Trainer 调用。

主要函数
    | 函数                 | 说明                                                                 |
    |----------------------|----------------------------------------------------------------------|
    | dist_setup           | 初始化 torch.distributed；非主进程屏蔽 print；设置 rank / barrier   |
    | concat_all_gather    | 各 rank 上对 tensor 做 all_gather 后沿 batch 维拼接（无梯度）        |

在 MoCo / ViT-MoCo 中的意义
    对比学习需要大量负样本；多卡时把各 GPU 上的 key 特征 gather 成「全局 batch」，
    等价于扩大负样本池，提升自监督表示学习效果（见 moco_v3.contrastive_loss）。
"""
import torch
import torch.distributed as dist
import builtins

__all__ = [
    "concat_all_gather",
    "dist_setup",
]

def dist_setup(ngpus_per_node, args):
    torch.multiprocessing.set_start_method('fork', force=True)
    # suppress printing if not master
    if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + args.gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        dist.barrier()

@torch.no_grad()
def concat_all_gather(tensor, distributed=True):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if distributed:
        dist.barrier()
        tensors_gather = [torch.ones_like(tensor)
            for _ in range(dist.get_world_size())]
        # print(f"World size: {dist.get_world_size()}")
        dist.all_gather(tensors_gather, tensor, async_op=False)

        output = torch.cat(tensors_gather, dim=0)
        return output
    else:
        return tensor