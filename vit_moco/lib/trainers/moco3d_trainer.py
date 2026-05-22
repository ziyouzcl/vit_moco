# -*- coding: utf-8 -*-
"""
@File: moco_3d_trainer.py
@Brief:
    3D MoCo v3 自监督预训练器。

@Description:
    本模块基于 BaseTrainer 封装 3D 医学影像的 MoCo v3 训练流程，
    适用于 CT/MRI 等三维体数据的自监督表征学习。训练时，Dataset
    需为同一样本生成两种增强视图，模型接收 image_q、image_k 和
    momentum 系数，并在 forward 内部计算对比学习损失。

@Main Features:
    1. 支持 3D MoCo v3 自监督预训练流程
    2. 支持 AMP 混合精度训练
    3. 支持单卡 / 多卡 DistributedDataParallel 训练
    4. 支持 LARS、AdamW、SGD 优化器
    5. 支持 warmup + cosine 学习率调度
    6. 支持 MoCo momentum encoder 动量余弦调度
    7. 支持 checkpoint 保存与断点续训

@Input:
    train_dataset_obj:
        每个样本应返回 (images, label)，其中 images 为两个增强视图：
        images[0] 和 images[1]。

@Output:
    训练得到的 3D encoder / MoCo 模型权重，可用于下游分类、预后预测
    或特征提取任务。

@Note:
    当前 pretrain 加载逻辑为预留接口，遇到 args.pretrain 会直接 raise。
    若需要加载已有权重，需要先完善 build_model() 中的加载逻辑。
"""
import math
import os
import time
import torch

# lib/utils/mod_from_moco.py — 训练日志小工具（MoCo 风格）:
#   AverageMeter: 计算并存储某个指标的当前值、累计和、计数以及平均值；
#   ProgressMeter: 管理多个 AverageMeter，结合 batch 序号与总 batch 数生成格式化进度字符串，
from vit_moco.lib.utils.mod_from_moco import AverageMeter, ProgressMeter

from .base_trainer import BaseTrainer
import timm.optim

__all__ = ['MoCo3DTrainer']

class MoCo3DTrainer(BaseTrainer):
    r"""
    MoCo V3 3D Trainer
    """
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.model_name = args.model_name
        self.scaler = torch.cuda.amp.GradScaler()

    def build_model(self):
        """
        创建 MoCo 模型并完成 GPU/DDP 封装。

        流程：
        1. 挂载 YAML 中的 MoCo_ViT（args.model_obj）。
        2. 若配置了 pretrain 路径：当前会直接 raise，下方加载逻辑尚未启用。
        3. loss_fn 设为 Identity：对比损失在 model(x1, x2, m) 内计算，训练循环不再包一层 loss。
        4. wrap_model()：上 GPU，必要时包 DistributedDataParallel，得到 wrapped_model 供 epoch_train 使用。
        """
        if self.model_name != 'Unknown' and self.model is None:
            args = self.args

            print(f"=> creating model {self.model_name} of arch {args.arch}")
            # 模型在配置解析阶段已构造（MONAI $call 等），此处只挂载引用
            self.model = args.model_obj

            # 预训练权重加载（预留 enc+dec / enc / dec；当前遇到 pretrain 即报错，下面代码不可达）
            if args.pretrain is not None and os.path.exists(args.pretrain):
                print(f"=> Start loading pretrained weights from {args.pretrain}")
                checkpoint = torch.load(args.pretrain, map_location='cpu')
                raise ValueError("=> Pretrain is not supported yet")
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                if args.pretrain_load == 'enc+dec':
                    msg = self.model.load_state_dict(state_dict, strict=False)
                elif args.pretrain_load == 'enc':
                    state_dict = {k[len("encoder."):]:v for k,v in state_dict.items() if k.startswith('encoder.')}
                    msg = self.model.encoder.load_state_dict(state_dict, strict=False)
                elif args.pretrain_load == 'dec':
                    state_dict = {k[len("decoder."):]:v for k,v in state_dict.items() if k.startswith('decoder.')}
                    msg = self.model.decoder.load_state_dict(state_dict, strict=False)
                else:
                    raise ValueError(f"=> Wrong pretrain_load: {args.pretrain_load}")
                self.model.encoder.head.weight.data.normal_(mean=0.0, std=0.01)
                self.model.encoder.head.bias.data.zero_()
                print(f'Loading messages: \n {msg}')
                print(f"=> Finish loading pretrained weights from {args.pretrain}")

            # 占位；真实 loss 由 MoCo forward 返回
            self.loss_fn = torch.nn.Identity()

            # 单卡 cuda 或多卡 DDP，训练时用 self.wrapped_model
            self.wrap_model()
        elif self.model_name == 'Unknown':
            raise ValueError("=> Model name is still unknown")
        else:
            raise ValueError("=> Model has been created. Do not create twice")

    def build_optimizer(self):
        """
        在 build_model / wrap_model 之后创建优化器。

        流程：
        1. get_parameter_groups()：按 bias、1D 参数及模型 no_weight_decay() 将参数分为「有/无 weight decay」两组。
        2. 按 batch 线性缩放 lr：lr *= batch_size / pretrain_batch_size（默认参考 batch=256，与 MoCo 大 batch 习惯一致）。
        3. 按 args.optimizer 选用 LARS / AdamW / SGD 之一；MoCo 预训练常用 LARS。
        """
        assert(self.model is not None and self.wrapped_model is not None), \
                "Model is not created and wrapped yet. Please create model first."
        print("=> creating optimizer")
        args = self.args

        # 基类分组：decay 组用 args.weight_decay，no_decay 组为 0
        optim_params = self.get_parameter_groups()

        # 与参考 batch 对齐的有效学习率（改的是 args.lr，后续 adjust_learning_rate 在此基础上调度）
        print(f"Changing learning rate to match batch size, lr *= batch_size / {args.get('pretrain_batch_size', 256)}: ", args.lr)
        args.lr = args.lr * args.batch_size / args.get("pretrain_batch_size", 256)

        if args.optimizer == 'lars':
            self.optimizer = timm.optim.LARS(optim_params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        elif args.optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(optim_params, lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(optim_params, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        else:
            raise ValueError(f"Unsupported optimizer {args.optimizer}")
        
    def build_dataloader(self):
        """
        构建训练 DataLoader（仅训练集，MoCo 阶段无验证集 dataloader）。

        要求 train_dataset_obj 的 __getitem__ 返回 (images, _)，其中 images 为长度 2 的列表/元组，
        即同一 3D 样本的两种增强视图，供 epoch_train 中 model(images[0], images[1], moco_m) 使用。
        多卡时用 DistributedSampler；batch_size / workers 可能在 wrap_model 中已按 GPU 数调整。
        """
        if self.dataloader is None:
            print("=> creating train dataloader")
            args = self.args

            # 由 YAML / get_conf 构造好的 Dataset，非本类内 new
            train_dataset = args.train_dataset_obj

            # 分布式：每卡不同子集，shuffle 由 sampler 负责，DataLoader 不再 shuffle
            if args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
            else:
                train_sampler = None

            self.dataloader = torch.utils.data.DataLoader(train_dataset,
                                                          batch_size=self.batch_size,
                                                          shuffle=(train_sampler is None),
                                                          num_workers=self.workers,
                                                          pin_memory=True,
                                                          sampler=train_sampler,
                                                          drop_last=True)  # 丢弃不足一整 batch，避免对比学习 batch 不齐
            self.iters_per_epoch = len(self.dataloader)  # 供学习率/动量按 step 调度与日志步数
        else:
            raise ValueError(f"Dataloader has been created. Do not create twice.")
        
    def run(self):
        """
        训练主循环：按 epoch 调用 epoch_train，并在 rank 0 上定期保存 checkpoint。

        MoCo 自监督阶段无验证集；是否保存 best 依赖 epoch_train 返回的 loss（当前实现未 return，save_best 时需注意）。
        """
        args = self.args
        # 断点续训时累计已走过的 iteration 数（传入 epoch_train，当前 epoch_train 内未使用）
        niters = args.start_epoch * self.iters_per_epoch

        # save_best 时用于比较的训练 loss 下界（越小越好）
        best_metric = torch.inf
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                # 每 epoch 换 shuffle 种子，保证各 rank 划分一致且每轮顺序不同
                self.dataloader.sampler.set_epoch(epoch)
                torch.distributed.barrier()

            # 训练一个 epoch；返回值赋给 loss，供 save_best 比较（见 epoch_train 说明）
            loss = self.epoch_train(epoch, niters)

            # 仅主进程或非 mp 分布式时写盘，避免多进程同时写同一文件
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
                # 每 save_freq 个 epoch 存一次定期 checkpoint
                if (epoch + 1) % args.save_freq == 0:
                    self.save_checkpoint(
                        {
                            'epoch': epoch + 1,
                            'arch': args.arch,
                            'state_dict': self.model.state_dict(),  # 未包 DDP 的 self.model，键无 module. 前缀（与 wrap 方式有关）
                            'optimizer' : self.optimizer.state_dict(),
                            'scaler': self.scaler.state_dict(),  # AMP GradScaler，续训需一并恢复
                        },
                        is_best=False,
                        filename=f'{args.ckpt_dir}/checkpoint_{epoch:04d}.pth.tar'
                    )
                    # 若开启 save_best 且本 epoch loss 更优，额外覆盖 best.pth.tar
                    if getattr(args, "save_best", False) and loss < best_metric:
                        best_metric = loss
                        self.save_checkpoint(
                            {
                                'epoch': epoch + 1,
                                'arch': args.arch,
                                'state_dict': self.model.state_dict(),
                                'optimizer' : self.optimizer.state_dict(),
                                'scaler': self.scaler.state_dict(),
                            },
                            is_best=False,
                            filename=f'{args.ckpt_dir}/best.pth.tar'
                        )

    def epoch_train(self, epoch, niters):
        """
        单个 epoch 的训练循环：双视图前向、对比损失反传、日志与 TensorBoard。

        参数:
            epoch: 当前 epoch 序号（从 0 或 start_epoch 起）。
            niters: 全局 iteration 偏移，预留续训/日志用，当前函数体内未使用。

        每个 batch:
            images[0]/images[1] 为同一样本两种增强；loss 由 MoCo_ViT.forward 内部计算。
        """
        args = self.args
        train_loader = self.dataloader
        model = self.wrapped_model  # 可能为 DDP 包装，forward 会走对比学习
        optimizer = self.optimizer
        scaler = self.scaler  # 混合精度梯度缩放
        loss_fn = self.loss_fn  # Identity，未参与 loss 计算

        # 终端进度条用的滑动平均表
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        learning_rates = AverageMeter('LR', ':.4e')
        losses = AverageMeter('Loss', ':.4e')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, learning_rates, losses],
            prefix="Epoch: [{}]".format(epoch))

        model.train()

        end = time.time()  # 用于统计 data_time / batch_time
        iters_per_epoch = len(train_loader)
        moco_m = args.moco_m  # 动量 encoder EMA 系数；moco_m_cos 时会在循环内被覆盖
        for i, (images, _) in enumerate(train_loader):
            # 标签 _ 不使用；双视图来自 Dataset + MultiTransforms
            if images[0].isnan().any() or images[1].isnan().any():
                print("images nan detected")
                continue  # 跳过本 batch，不更新参数

            data_time.update(time.time() - end)

            # 按「小数 epoch」调度：warmup + 余弦退火（见 adjust_learning_rate）
            lr = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, args)
            learning_rates.update(lr)
            if args.moco_m_cos:
                # 动量系数随训练进度余弦变化（见 adjust_moco_momentum）
                moco_m = adjust_moco_momentum(epoch + i / iters_per_epoch, args)

            if args.gpu is not None:
                images[0] = images[0].cuda(args.gpu, non_blocking=True)
                images[1] = images[1].cuda(args.gpu, non_blocking=True)

            # 前向：在线 encoder+predictor vs 动量 encoder，返回标量对比损失
            with torch.cuda.amp.autocast(True):
                loss = model(images[0], images[1], moco_m)

            if not torch.isfinite(loss).all():
                print("loss nan/inf detected, skip batch")
                continue

            losses.update(loss.item(), images[0].size(0))
            if args.rank == 0:
                # 需 finetuning_main 中已创建 args.summary_writer
                args.summary_writer.add_scalar("loss", loss.item(), epoch * iters_per_epoch + i)
                args.summary_writer.add_scalar("epoch", epoch, epoch * iters_per_epoch + i)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
                if args.rank == 0:
                    args.summary_writer.add_text('info', progress.get_display(i), epoch * iters_per_epoch + i)

    def resume(self):
        """
        从 args.resume 路径加载 checkpoint，恢复模型与优化器，并设置 args.start_epoch。

        注意：未恢复 scaler；若需完整 AMP 续训需在调用方扩展加载 scaler.state_dict。
        """
        args = self.args
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


def adjust_learning_rate(optimizer, epoch, args):
    """
    按 iteration 更新优化器学习率（warmup 后半周期余弦衰减）。

    参数 epoch 可为小数，如 epoch_train 传入的 epoch + i/iters_per_epoch，实现逐 step 调度。
    warmup 阶段: lr 从 0 线性增至 args.lr；之后: lr = args.lr * 0.5 * (1 + cos(...))。
    """
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_moco_momentum(epoch, args):
    """
    按训练进度调整 MoCo 动量 encoder 的 EMA 系数 m（余弦从 1 走向 args.moco_m）。

    epoch 同样可为小数；仅在 args.moco_m_cos 为真时由 epoch_train 每 step 调用。
    """
    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - args.moco_m)
    return m